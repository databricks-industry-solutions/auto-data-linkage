package com.databricks.industry.solutions.arc.expressions

import com.databricks.industry.solutions.arc.{functions, ARC}
import org.apache.spark.sql.QueryTest
import org.apache.spark.sql.test.SharedSparkSession
import org.scalatest.matchers.must.Matchers.{be, noException}

import java.util

class ARC_Test extends QueryTest with SharedSparkSession {

    test("ARC_CombinatorialCountAgg expression") {

        import scala.collection.JavaConverters._

        val values = Seq(
            (1, "a", "b", "c", "d"),
            (2, "a", "b", "c", "d"),
            (3, "a", "b", "c", "d"),
            (4, "g", "h1", ";", "a0"),
            (5, "g", "h2", ";", "a1"),
            (6, "g", "h3", ";", "a2"),
            (7, "g", "h1", ";", "a2"),
            (9, "g", "h2", ";", "a1")
        )

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

        noException should be thrownBy ARC.generateBlockingRules(testData, 3, 4, Seq("a", "b", "c", "d")).limit(1000).collect()

        noException should be thrownBy functions.arc_generate_blocking_rules(testData, 3, 4, "a", "b", "c", "d").limit(1000).collect()

        val asJavaList = new util.ArrayList[String]()
        asJavaList.addAll(Seq("a", "b", "c", "d").asJava)

        noException should be thrownBy functions.arc_generate_blocking_rules(testData, 3, 4, asJavaList).limit(1000).collect()

    }

}
