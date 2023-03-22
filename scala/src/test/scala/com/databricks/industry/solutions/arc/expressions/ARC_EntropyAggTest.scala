package com.databricks.industry.solutions.arc.expressions

import org.apache.spark.sql.QueryTest
import org.apache.spark.sql.test.SharedSparkSession

class ARC_EntropyAggTest extends QueryTest with SharedSparkSession with ARC_EntropyAggBehaviors {

    test("ARC_EntropyAgg expression") { testEntropyAgg() }

    test("ARC_EntropyAgg values") { testEntropyAggValues() }

}
