package com.databricks.industry.solutions.arc

import com.databricks.industry.solutions.arc.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._

object ARC {

    def generateCountLookup(df: DataFrame, k: Int, attributes: Seq[String]): DataFrame = {
        val window = Window.partitionBy(col("rule"))
        df
            .select(
              arc_combinatorial_count_agg(k, attributes: _*) as "combinations"
            )
            .select(
              explode(
                col("combinations")
              ).as(Seq("key", "count"))
            )
            .withColumn("k", col("count") * col("count"))
            .withColumn("rule", split(col("key"), ";").getItem(0))
            .repartition(col("rule"))
            // low_k, med_k and high_k are the 25th, 50th and 75th percentiles of k
            // these are used to trim the distribution of k and compute the trimmed mean
            .withColumn("percentiles", percentile_approx(col("k"), array(lit(0.1), lit(0.5), lit(0.9)), lit(1000)).over(window))
            .withColumn(
              "adjusted_k",
              when(expr("k > percentiles[0] and k < percentiles[2]"), col("k")).otherwise(col("percentiles").getItem(1))
            )
            .groupBy(col("rule"))
            .agg(
              count(lit(1)).as("n"),
              floor(avg(col("adjusted_k"))).cast("long").as("avg_k")
            )
            .select(
              col("rule"),
              struct(
                col("n"), // number of blocks
                col("avg_k") // trimmed average squared count
              ).as("rule_stats")
            )
    }

    def generateRules(df: DataFrame, n: Int, countMap: Map[String, (Long, Long)]): DataFrame = {
        val partials = df
            .select(
              arc_generate_combinations(n, col("rules")) as "combinations",
              col("rules")
            )

        val combinations =
            if (n > 1) {
                partials
                    .select(
                      arc_generate_partial_combinations(n - 2, col("combinations"), col("rules")) as "combinations"
                    )
            } else {
                partials
            }

        combinations
            .distinct()
            .select(
              arc_to_splink_rule(col("combinations")) as "splink_rule",
              arc_estimate_squared_count_or(col("combinations"), countMap) as "rule_squared_count"
            )
    }

    def generateORRules(countMap: Map[String, (Long, Long)], k: Int): DataFrame = {
        val spark = SparkSession.builder().getOrCreate()
        val baseDf = spark
            .createDataFrame(
              Seq((1, countMap.keys.toSeq))
            )
            .toDF("id", "rules")

        val dfN1 = generateRules(baseDf, 1, countMap)

        if (k > 1) {
            (2 to k).foldLeft(dfN1)((df, i) => df union generateRules(baseDf, i, countMap)).distinct()
        } else {
            dfN1.distinct()
        }

    }

    def generateBlockingRules(df: DataFrame, n: Int, k: Int, attributes: Seq[String]): DataFrame = {
        val spark = SparkSession.builder().getOrCreate()
        import spark.implicits._

        val countMap = generateCountLookup(df, n, attributes)
            .as[(String, (Long, Long))]
            .collect()
            .toMap

        generateORRules(countMap, k)
    }

}
