package com.databricks.industry.solutions.arc.expressions

import com.databricks.industry.solutions.arc.functions.arc_entropy_agg
import org.apache.spark.sql.QueryTest
import org.scalatest.matchers.must.Matchers.{be, convertToAnyMustWrapper, noException}

trait ARC_EntropyAggBehaviors extends QueryTest {

    def testEntropyAgg(): Unit = {
        spark.sparkContext.setLogLevel("FATAL")

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

        noException should be thrownBy testData.select(
          arc_entropy_agg("a", "b", "c", "d").as("entropies")
        )

    }

    def testEntropyAggValues(): Unit = {
        spark.sparkContext.setLogLevel("FATAL")

        val testData = spark
            .createDataFrame(
              Seq(
                (1, "a", "b", "c", "d"),
                (2, "a", "b", "c", "c"),
                (3, "a", "b", "c", "g"),
                (4, "g", "h1", ";", "a0"),
                (5, "g", "h2", "+", "a1"),
                (6, "g", "h3", "/", "a2"),
                (7, "g", "h1", ";", "a2"),
                (9, "g", "h2", "1", "a1")
              )
            )
            .toDF("id", "a", "b", "c", "d")

        val result = testData
            .select(
              arc_entropy_agg("a", "b", "c", "d").as("entropies")
            )
            .first()

        result.getAs[Map[String, Double]]("entropies") must be(
          Map(
            "a" -> (-3.0 / 8 * math.log(3 / 8.0) - 5.0 / 8.0 * math.log(5.0 / 8.0)),
            "b" -> (
              -3.0 / 8 * math.log(3.0 / 8.0)
                  - 2.0 / 8.0 * math.log(2.0 / 8.0)
                  - 2.0 / 8.0 * math.log(2.0 / 8.0)
                  - 1.0 / 8.0 * math.log(1.0 / 8.0)
            ) / math.log(4.0),
            "c" -> (
              -3.0 / 8 * math.log(3.0 / 8.0)
                  - 2.0 / 8.0 * math.log(2.0 / 8.0)
                  - 1.0 / 8.0 * math.log(1.0 / 8.0)
                  - 1.0 / 8.0 * math.log(1.0 / 8.0)
                  - 1.0 / 8.0 * math.log(1.0 / 8.0)
            ) / math.log(5.0),
            "d" -> (
              -1.0 / 8 * math.log(1.0 / 8.0)
                  - 1.0 / 8.0 * math.log(1.0 / 8.0)
                  - 1.0 / 8.0 * math.log(1.0 / 8.0)
                  - 1.0 / 8.0 * math.log(1.0 / 8.0)
                  - 2.0 / 8.0 * math.log(2.0 / 8.0)
                  - 2.0 / 8.0 * math.log(2.0 / 8.0)
                ) / math.log(6.0)
          )
        )
    }

}
