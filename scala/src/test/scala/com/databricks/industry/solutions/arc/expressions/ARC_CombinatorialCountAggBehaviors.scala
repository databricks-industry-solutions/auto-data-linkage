package com.databricks.industry.solutions.arc.expressions

import com.databricks.industry.solutions.arc.functions.{arc_combinatorial_count_agg, arc_merge_count_map}
import org.apache.spark.sql.QueryTest
import org.apache.spark.sql.functions.{col, lit}
import org.scalatest.matchers.must.Matchers.{contain, convertToAnyMustWrapper}

trait ARC_CombinatorialCountAggBehaviors extends QueryTest {

    def testCombinatorialCountAgg(): Unit = {
        spark.sparkContext.setLogLevel("FATAL")
        val sc = spark
        import sc.implicits._

        val testData = spark
            .createDataFrame(
              Seq(
                (1, "a", "b", "c", "d"),
                (2, "a", "b", "c", "d"),
                (3, "a", "b", "c", "d"),
                (4, "g", "h1", ";", "a0"),
                (5, "g", "h2", ";", "a1"),
                (6, "g", "h3", ";", "a2"),
                (7, "g", "h1", ";", "a2"),
                (9, "g", "h2", ";", "a1")
              )
            )
            .toDF("id", "a", "b", "c", "d")

        val result = testData
            .select(
              arc_combinatorial_count_agg("a", "b", "c", "d").as("combinations")
            )
            .as[Map[String, Long]]
            .take(1)
            .head

        val expected = Map(
          "a AND b;a,b" -> 3L,
          "a AND c;a,c" -> 3L,
          "a AND d;a,d" -> 3L,
          "b AND c;b,c" -> 3L,
          "b AND d;b,d" -> 3L,
          "c AND d;c,d" -> 3L,
          "a AND b;g,h1" -> 2L,
          "a AND b;g,h2" -> 2L,
          "a AND b;g,h3" -> 1L,
          "a AND c;g,;" -> 5L,
          "a AND d;g,a0" -> 1L,
          "a AND d;g,a1" -> 2L,
          "a AND d;g,a2" -> 2L,
          "b AND c;h1,;" -> 2L,
          "b AND d;h1,a0" -> 1L,
          "b AND d;h1,a2" -> 1L,
          "b AND c;h2,;" -> 2L,
          "b AND d;h2,a1" -> 2L,
          "b AND c;h3,;" -> 1L,
          "b AND d;h3,a2" -> 1L,
          "c AND d;;,a0" -> 1L,
          "c AND d;;,a1" -> 2L,
          "c AND d;;,a2" -> 2L
        )

        result must contain theSameElementsAs expected

    }

    def testMergeCountMapAgg(): Unit = {
        spark.sparkContext.setLogLevel("FATAL")
        val sc = spark
        import sc.implicits._

        val testData = spark
            .createDataFrame(
              Seq(
                (1, "a", "b", "c", "d"),
                (2, "a", "b", "c", "d"),
                (3, "a", "b", "c", "d"),
                (4, "g", "h1", ";", "a0"),
                (5, "g", "h2", ";", "a1"),
                (6, "g", "h3", ";", "a2"),
                (7, "g", "h1", ";", "a2"),
                (9, "g", "h2", ";", "a1")
              )
            )
            .toDF("id", "a", "b", "c", "d")

        val result = testData
            .groupBy(col("id") % 2 as "group")
            .agg(arc_combinatorial_count_agg("a", "b", "c", "d").as("combinations"))
            .groupBy(lit(1) as "group")
            .agg(
              arc_merge_count_map(col("combinations")).as("combinations")
            )
            .select("combinations")
            .as[Map[String, Long]]
            .take(1)
            .head

        val expected = testData
            .select(
              arc_combinatorial_count_agg("a", "b", "c", "d").as("combinations")
            )
            .select("combinations")
            .as[Map[String, Long]]
            .take(1)
            .head

        result must contain theSameElementsAs expected
    }

}
