package com.databricks.industry.solutions.arc

import com.databricks.industry.solutions.arc.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

object ARC {

    def generateCountLookup(df: DataFrame, k: Int, attributes: Seq[String]): DataFrame = {
        df
            .select(
              arc_combinatorial_count_agg(k, attributes: _*) as "combinations"
            )
            .select(
              explode(
                col("combinations")
              ).as(Seq("key", "count"))
            )
            .withColumn("count_squared", col("count") * col("count"))
            .groupBy(
              split(col("key"), ";").getItem(0).as("rule")
            )
            .agg(
              sum(col("count_squared")).as("sum_count_squared"),
              max(col("count_squared")).as("max_count_squared")
            )
            .select(
              col("rule"),
              struct(
                col("sum_count_squared"),
                col("max_count_squared")
              ).as("rule_stats")
            )
    }

    def generateORRules(countMap: Map[String, (Long, Long)], k: Int): DataFrame = {
        val spark = SparkSession.builder().getOrCreate()
        val baseDf = spark
            .createDataFrame(
              Seq((1, countMap.keys.toSeq))
            )
            .toDF("id", "rules")

        def generateRules(df: DataFrame, n: Int) = {
            val nPartitions = spark.conf.get("spark.sql.shuffle.partitions").toInt
            df
                .select(
                  arc_generate_combinations(2, col("rules")) as "combinations",
                  col("rules")
                )
                .repartition(nPartitions, col("combinations"))
                .select(
                  arc_generate_partial_combinations(n - 2, col("combinations"), col("rules")) as "combinations"
                )
                .distinct()
                .select(
                  arc_to_splink_rule(col("combinations")) as "splink_rule",
                  arc_estimate_squared_count_or(col("combinations"), countMap) as "rule_squared_count"
                )
        }

        val dfN1 = generateRules(baseDf, 1)

        if (k > 1) {
            (2 to k).foldLeft(dfN1)((df, i) => df union generateRules(baseDf, i)).distinct()
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
