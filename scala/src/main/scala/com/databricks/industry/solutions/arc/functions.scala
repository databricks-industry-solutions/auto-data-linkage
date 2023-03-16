package com.databricks.industry.solutions.arc

import com.databricks.industry.solutions.arc.expressions._
import org.apache.spark.sql.Column

import scala.collection.JavaConverters.asScalaBufferConverter

object functions {

    def arc_combinatorial_count_agg(cols: String*): Column = {
        val exprs = cols.map(cn => new Column(cn).expr)
        new Column(ARC_CombinatorialCountAgg(exprs, cols).toAggregateExpression())
    }

    def arc_combinatorial_count_agg(nCombination: Int, cols: String*): Column = {
        val exprs = cols.map(cn => new Column(cn).expr)
        new Column(ARC_CombinatorialCountAgg(exprs, cols, nCombination).toAggregateExpression())
    }

    def arc_combinatorial_count_agg(nCombination: Int, cols: java.util.ArrayList[String]): Column = {
        arc_combinatorial_count_agg(nCombination, cols.asScala: _*)
    }

    def arc_entropy_agg(cols: String*): Column = {
        val exprs = cols.map(cn => new Column(cn).expr)
        new Column(ARC_EntropyAggExpression(exprs, cols).toAggregateExpression())
    }

    def arc_entropy_agg(cols: java.util.ArrayList[String]): Column = {
        arc_entropy_agg(cols.asScala: _*)
    }

    def arc_merge_count_map(counter_map: Column): Column = {
        new Column(ARC_MergeCountMapAgg(counter_map.expr).toAggregateExpression())
    }

}
