package com.databricks.industry.solutions.arc.expressions

import com.databricks.industry.solutions.arc.functions.arc_generate_combinations
import org.apache.spark.sql.QueryTest
import org.scalatest.matchers.must.Matchers.{contain, convertToAnyMustWrapper}

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

}
