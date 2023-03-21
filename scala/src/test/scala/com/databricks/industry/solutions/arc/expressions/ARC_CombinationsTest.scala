package com.databricks.industry.solutions.arc.expressions

import org.apache.spark.sql.QueryTest
import org.apache.spark.sql.test.SharedSparkSession

class ARC_CombinationsTest extends QueryTest with SharedSparkSession with ARC_CombinationsBehaviors {

    test("ARC_CombinatorialCountAgg expression") { testCombinations() }

}
