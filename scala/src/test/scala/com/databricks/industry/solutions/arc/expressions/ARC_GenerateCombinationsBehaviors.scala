package com.databricks.industry.solutions.arc.expressions

import com.databricks.industry.solutions.arc.functions.{arc_generate_combinations, arc_generate_partial_combinations}
import org.apache.spark.sql.QueryTest
import org.apache.spark.sql.functions.col
import org.scalatest.matchers.must.Matchers.{contain, convertToAnyMustWrapper}
import org.scalatest.matchers.should.Matchers.convertToAnyShouldWrapper

trait ARC_GenerateCombinationsBehaviors extends QueryTest {

    def testGenerateCombinations(): Unit = {
        spark.sparkContext.setLogLevel("FATAL")
        val sc = spark
        import sc.implicits._

        val testData = spark
            .createDataFrame(
              Seq(
                (1, Seq("a", "b", "c", "d"))
              )
            )
            .toDF("id", "columns")

        val result = testData
            .select(
              arc_generate_combinations(2, "a", "b", "c", "d").as("combinations")
            )
            .as[Seq[String]]
            .collect()
            .toSeq

        val expected = Seq(
          Seq("a", "b"),
          Seq("a", "c"),
          Seq("a", "d"),
          Seq("b", "c"),
          Seq("b", "d"),
          Seq("c", "d")
        )

        result must contain theSameElementsAs expected

    }

    def testGeneratePartialCombinations(): Unit = {
        spark.sparkContext.setLogLevel("FATAL")
        val sc = spark
        import sc.implicits._

        val values = Seq("a", "b", "c", "d", "e", "f", "g", "h", "i", "j")
        val testData = spark
            .createDataFrame(
              Seq(
                (1, values)
              )
            )
            .toDF("id", "columns")

        val result = testData
            .withColumn(
              "combinations",
              arc_generate_combinations(2, col("columns"))
            )
            .select(
              arc_generate_partial_combinations(3, col("combinations"), col("columns")).as("combinations")
            )
            .distinct()
            .as[Seq[String]]
            .collect()
            .toSeq

        val n = values.combinations(5).size

        result.size shouldEqual n

    }

}
