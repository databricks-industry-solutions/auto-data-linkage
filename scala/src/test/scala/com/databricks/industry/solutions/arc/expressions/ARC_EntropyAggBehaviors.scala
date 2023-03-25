package com.databricks.industry.solutions.arc.expressions

import com.databricks.industry.solutions.arc.expressions.base._
import com.databricks.industry.solutions.arc.functions.arc_entropy_agg
import org.apache.spark.sql.QueryTest
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.BoundReference
import org.apache.spark.sql.catalyst.util.MapData
import org.apache.spark.sql.types.StringType
import org.apache.spark.unsafe.types.UTF8String
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

        result.getAs[Map[String, Double]]("entropies").mapValues(v => math.round(1000 * v)) must be(
          Map(
            "a" -> (-3.0 / 8 * math.log(3 / 8.0) - 5.0 / 8.0 * math.log(5.0 / 8.0)) / math.log(2),
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
          ).mapValues(v => math.round(1000 * v))
        )
    }

    def testExpressionMethods(): Unit = {

        val testData = spark
            .createDataFrame(
              Seq(
                ("a", "b", "c", "d"),
                ("a", "b", "c", "c"),
                ("a", "b", "c", "g"),
                ("g", "h1", ";", "a0"),
                ("g", "h2", "+", "a1"),
                ("g", "h3", "/", "a2"),
                ("g", "h1", ";", "a2"),
                ("g", "h2", "1", "a1")
              )
            )
            .toDF("a", "b", "c", "d")

        val columns = testData.columns

        val expr = ARC_EntropyAggExpression(
          Seq(
            BoundReference(0, StringType, nullable = true),
            BoundReference(1, StringType, nullable = true),
            BoundReference(2, StringType, nullable = true),
            BoundReference(3, StringType, nullable = true)
          ),
          columns
        )

        val expectedCounter = columns.map(k => k -> CountAccumulatorMap(Map.empty[String, Long])).toMap

        expr.createAggregationBuffer().counter must be(expectedCounter)

        val nextRow1 = InternalRow.fromSeq(Seq("a", "b", "c", "d").map(UTF8String.fromString))
        val nextRow2 = InternalRow.fromSeq(Seq("a", "b", "c", "c").map(UTF8String.fromString))
        val nextRow3 = InternalRow.fromSeq(Seq("a", "b", "c", "g").map(UTF8String.fromString))
        val nextRow4 = InternalRow.fromSeq(Seq("g", "h1", ";", "a0").map(UTF8String.fromString))
        val nextRow5 = InternalRow.fromSeq(Seq("g", "h2", "+", "a1").map(UTF8String.fromString))

        val updated1 = expr.update(expr.createAggregationBuffer(), nextRow1)
        val updated2 = expr.update(updated1, nextRow2)
        val updated3 = expr.update(updated2, nextRow3)
        val updated4 = expr.update(updated3, nextRow4)
        val updated5 = expr.update(updated4, nextRow5)
        val updated6 = expr.update(updated5, nextRow1)
        val updated7 = expr.update(updated6, nextRow2)

        updated1.counter must be(
          Map(
            "a" -> CountAccumulatorMap(Map("a" -> 1L)),
            "b" -> CountAccumulatorMap(Map("b" -> 1L)),
            "c" -> CountAccumulatorMap(Map("c" -> 1L)),
            "d" -> CountAccumulatorMap(Map("d" -> 1L))
          )
        )

        updated2.counter must be(
          Map(
            "a" -> CountAccumulatorMap(Map("a" -> 2L)),
            "b" -> CountAccumulatorMap(Map("b" -> 2L)),
            "c" -> CountAccumulatorMap(Map("c" -> 2L)),
            "d" -> CountAccumulatorMap(Map("c" -> 1L, "d" -> 1L))
          )
        )

        updated7.counter must be(
          Map(
            "a" -> CountAccumulatorMap(Map("a" -> 5L, "g" -> 2L)),
            "b" -> CountAccumulatorMap(Map("b" -> 5L, "h1" -> 1L, "h2" -> 1L)),
            "c" -> CountAccumulatorMap(Map("c" -> 5L, "+" -> 1L, ";" -> 1L)),
            "d" -> CountAccumulatorMap(Map("c" -> 2L, "d" -> 2L, "g" -> 1L, "a0" -> 1L, "a1" -> 1L))
          )
        )

        val updated8 = expr.merge(updated7, updated2)
        val updated9 = expr.merge(updated8, updated7)

        updated8.counter must be(
          Map(
            "a" -> CountAccumulatorMap(Map("a" -> 7L, "g" -> 2L)),
            "b" -> CountAccumulatorMap(Map("b" -> 7L, "h1" -> 1L, "h2" -> 1L)),
            "c" -> CountAccumulatorMap(Map("c" -> 7L, "+" -> 1L, ";" -> 1L)),
            "d" -> CountAccumulatorMap(Map("c" -> 3L, "d" -> 3L, "g" -> 1L, "a0" -> 1L, "a1" -> 1L))
          )
        )

        updated9.counter must be(
          Map(
            "a" -> CountAccumulatorMap(Map("a" -> 12L, "g" -> 4L)),
            "b" -> CountAccumulatorMap(Map("b" -> 12L, "h1" -> 2L, "h2" -> 2L)),
            "c" -> CountAccumulatorMap(Map("c" -> 12L, "+" -> 2L, ";" -> 2L)),
            "d" -> CountAccumulatorMap(Map("c" -> 5L, "d" -> 5L, "g" -> 2L, "a0" -> 2L, "a1" -> 2L))
          )
        )

        val ser1 = expr.serialize(updated9)
        val des1 = expr.deserialize(ser1)

        des1.counter must be(updated9.counter)

        val longMap = EntropyCountAccumulatorMap(
          Map(
            "a" -> CountAccumulatorMap(
              Map(
                "28392.0" -> 15L,
                "28152.0" -> 190L,
                "27317" -> 2L,
                "28316" -> 3L,
                "27990" -> 1L,
                "27884" -> 9L,
                "28117" -> 34L,
                "28197" -> 1L,
                "28334" -> 27L,
                "28135" -> 5L,
                "28675" -> 8L,
                "28503" -> 1L,
                "27s96" -> 1L,
                "27248" -> 5L,
                "28021.0" -> 99L,
                "27429" -> 1L,
                "27890" -> 5L,
                "28524" -> 2L,
                "28463" -> 6L,
                "28102" -> 3L,
                "2726s" -> 4L,
                "27339" -> 2L,
                "28449" -> 3L,
                "28039" -> 4L,
                "27541" -> 9L,
                "28405.0" -> 332L,
                "28610" -> 5L,
                "27841" -> 1L,
                "27020" -> 3L,
                "28088.0" -> 26L,
                "27701" -> 21L,
                "278s8" -> 13L,
                "27560" -> 22L,
                "28056" -> 26L,
                "27o55" -> 2L,
                "287|3" -> 2L,
                "28628.0" -> 11L
              )
            )
          )
        )

        Math.round(1000 * expr.eval(longMap).asInstanceOf[MapData].valueArray().toDoubleArray().head) must be(623)

    }

}
