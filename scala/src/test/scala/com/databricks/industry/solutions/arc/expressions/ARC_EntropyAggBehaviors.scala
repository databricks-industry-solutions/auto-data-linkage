package com.databricks.industry.solutions.arc.expressions

import com.databricks.industry.solutions.arc.expressions.base._
import com.databricks.industry.solutions.arc.functions.arc_entropy_agg
import org.apache.spark.sql.QueryTest
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.BoundReference
import org.apache.spark.sql.catalyst.util.MapData
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StringType
import org.apache.spark.unsafe.types.UTF8String
import org.scalactic.Tolerance.convertNumericToPlusOrMinusWrapper
import org.scalatest.matchers.must.Matchers.{be, convertToAnyMustWrapper, noException}
import org.scalatest.matchers.should.Matchers.convertToAnyShouldWrapper

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

        val sc = spark
        import sc.implicits._

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
            "a" -> BoundReference(0, StringType, nullable = true),
            "b" -> BoundReference(1, StringType, nullable = true),
            "c" -> BoundReference(2, StringType, nullable = true),
            "d" -> BoundReference(3, StringType, nullable = true)
          ).toMap
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

        val counts = Seq(3, 1, 2, 2, 1, 5, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1, 13, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 4, 8, 1, 1, 2, 2, 3, 1, 4,
          1, 3, 1, 9, 3, 10, 2, 3, 3, 1, 6, 1, 2, 2, 1, 2, 4, 1, 1, 2, 7, 4, 3, 18, 1, 1, 1, 1, 2, 6, 9, 5, 1, 1, 5, 2, 2, 12, 3, 1, 1, 1,
          1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 4, 3, 1, 2, 3, 2, 2, 2, 6, 2, 6, 1, 1, 1, 1, 8, 4, 2, 1, 1, 1, 8, 1, 1, 15, 4, 1, 1, 1, 2, 3, 4, 1,
          4, 1, 2, 2, 2, 2, 1, 3, 8, 4, 5, 2, 1, 3, 1, 1, 8, 1, 2, 3, 1, 4, 1, 2, 1, 1, 1, 2, 9, 2, 8, 2, 5, 1, 1, 2, 1, 2, 5, 1, 1, 2, 3,
          2, 1, 1, 6, 1, 2, 2, 6, 1, 7, 2, 1, 1, 5, 2, 3, 1, 4, 2, 2, 8, 6, 3, 5, 5, 2, 1, 1, 8, 11, 1, 2, 1, 13, 10, 2, 2, 4, 1, 1, 8, 1,
          2, 2, 2, 2, 2, 3, 1, 1, 2, 2, 8, 8, 5, 2, 2, 1, 1, 1, 1, 4, 1, 1, 1, 2, 1, 2, 1, 1, 5, 1, 2, 1, 1, 1, 1, 1, 1, 7, 1, 1, 1, 1, 1,
          1, 6, 3, 4, 1, 2, 1, 3, 3, 1, 1, 3, 1, 2, 1, 16, 1, 1, 2, 1, 4, 1, 1, 13, 8, 1, 1, 1, 3, 1, 2, 1, 1, 1, 7, 2, 1, 2, 2, 1, 1, 8, 2,
          2, 11, 2, 6, 6, 2, 1, 1, 11, 8, 1, 5, 6, 1, 2, 3, 1, 1, 1, 3, 2, 5, 2, 1, 1, 1, 1, 6, 5, 5, 1, 1, 2, 3, 1, 1, 3, 1, 1, 1, 1, 2, 1,
          4, 1, 1, 1, 1, 1, 1, 1, 7, 1, 2, 1, 1, 8, 9, 1, 5, 2, 1, 2, 1, 1, 1, 2, 2, 5, 1, 2, 1, 2, 9, 2, 2, 4, 1, 1, 8, 1, 2, 5, 4, 1, 16,
          2, 3, 1, 2, 2, 1, 2, 2, 1, 2, 3, 2, 4, 7, 1, 11, 4, 1, 3, 6, 11, 4, 1, 2, 12, 1, 10, 1, 1, 1, 6, 3, 3, 1, 14, 2, 8, 1, 1, 4, 1, 2,
          1, 6, 2, 1, 11, 2, 1, 1, 4, 1, 1, 1, 2, 1, 1, 5, 6, 9, 2, 2, 2, 1, 2, 1, 3, 2, 2, 1, 4, 2, 10, 2, 1, 10, 1, 1, 1, 2, 6, 1, 7, 7,
          1, 7, 1, 2, 1, 15, 3, 1, 1, 1, 2, 1, 1, 7, 4, 1, 1, 8, 1, 1, 2, 5, 5, 3, 4, 6, 2, 1, 3, 2, 4, 1, 2, 3, 2, 1, 1, 3, 1, 1, 1, 1, 2,
          1, 3, 2, 5, 2, 2, 1, 6, 1, 1, 4, 3, 1, 1, 13, 1, 3, 1, 2, 7, 4, 4, 3, 1, 1, 4, 2, 1, 2, 1, 1, 4, 1, 2, 1, 2, 1, 8, 1, 2, 2, 1, 1,
          7, 1, 2, 2, 9, 7, 2, 3, 5, 4, 1, 1, 5, 1, 1, 2, 4, 6, 1, 2, 1, 11, 1, 1, 2, 1, 3, 6, 12, 2, 2, 1, 1, 4, 2, 1, 1, 5, 2, 1, 4, 13,
          6, 2, 2, 1, 1, 11, 3, 2, 1, 8, 6, 1, 5, 3, 1, 1, 9, 4, 1, 1, 1, 1, 2, 4, 2, 1, 4, 1, 1, 15, 2, 2, 6, 1, 6, 12, 1, 21, 2, 2, 1, 3,
          6, 1, 1, 1, 3, 1, 3, 4, 5, 1, 1, 2, 1, 1, 1, 7, 4, 2, 2, 2, 1, 1, 2, 22, 8, 3, 1, 6, 1, 1, 1, 7, 1, 1, 1, 1, 2, 3, 8, 1, 1, 2, 10,
          13, 6, 11, 4, 1, 1, 2, 1, 1, 14, 2, 8, 10, 4, 1, 19, 1, 1, 1, 8, 8, 1, 1, 1, 2, 1, 1, 2, 6, 1, 7, 1, 2, 1, 1, 1, 9, 1, 1, 2, 1, 1,
          3, 1, 3, 1, 14, 2, 10, 2, 1, 1, 1, 2, 6, 14, 5, 1, 1, 1, 5, 1, 3, 2, 2, 9, 1, 1, 5, 2, 2, 1, 1, 1, 5, 1, 8, 6, 2, 2, 1, 1, 4, 1,
          2, 4, 3, 4, 1, 3, 2, 1, 2, 7, 14, 1, 7, 1, 8, 1, 2, 4, 1, 2, 2, 1, 1, 14, 9, 5, 7, 6, 9, 1, 1, 1, 1, 18, 2, 7, 4, 4, 6, 1, 2, 2,
          2, 1, 1, 3, 1, 1, 1, 1, 2, 3, 5, 1, 5, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 5, 6, 1, 1, 2, 7, 2, 3, 1, 1, 1, 7,
          4, 1, 1, 13, 1, 4, 3, 1, 1, 2, 1, 8, 7, 1, 1, 4, 12, 4, 9, 11, 2, 6, 1, 1, 3, 4, 2, 1, 8, 4, 3, 3, 9, 1, 2, 7, 4, 3, 1, 2, 4, 3,
          1, 2, 3, 1, 2, 1, 6, 18, 6, 1, 6, 1, 1, 6, 4, 4, 4, 8, 7, 1, 2, 3, 4, 6, 1, 5, 1, 1, 4, 1, 3, 4, 3, 2, 3, 1, 1, 1, 5, 4, 1, 1, 2,
          2, 1, 1, 1, 1)

        val df = spark
            .createDataFrame(Seq((1, counts)))
            .toDF("id", "k")

        val generateUDF = udf((n: Int) => {
            (0 until n).toArray
        })

        val testData2 = df
            .select(
              explode(col("k")).alias("k")
            )
            .withColumn(
              "id",
              monotonically_increasing_id()
            )
            .withColumn(
              "k",
              generateUDF(col("k"))
            )
            .withColumn(
              "k",
              explode(col("k"))
            )

        val entropy = counts
            .map(x => x.toDouble / counts.sum)
            .map(x => -x * math.log(x))
            .sum / math.log(counts.size)

        testData2
            .select(
              arc_entropy_agg(counts.size, "id")
            )
            .as[Map[String, Double]]
            .collect()
            .head("id") should be(entropy +- 0.0001)

    }

}
