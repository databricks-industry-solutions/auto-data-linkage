package com.databricks.industry.solutions.arc.expressions

import org.apache.spark.sql.QueryTest
import org.apache.spark.sql.test.SharedSparkSession

class ARC_CombinatorialCountAggTest extends QueryTest with SharedSparkSession with ARC_CombinatorialCountAggBehaviors {

    test("ARC_CombinatorialCountAgg expression") { testCombinatorialCountAgg() }

    test("ARC_MergeCountMapAgg expression") { testMergeCountMapAgg() }

}
