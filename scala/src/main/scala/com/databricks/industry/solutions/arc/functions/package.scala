package com.databricks.industry.solutions.arc

import com.databricks.industry.solutions.arc.expressions.{ARC_CombinatorialCountAgg, ARC_EntropyAggExpression, ARC_MergeCountMapAgg}
import org.apache.spark.sql.Column

package object functions {

    def arc_combinatorial_count_agg(cols: String*): Column = {
        val exprs = cols.map(cn => new Column(cn).expr)
        new Column(ARC_CombinatorialCountAgg(exprs, cols).toAggregateExpression())
    }

    def arc_combinatorial_count_agg(nCombination: Int, cols: String*): Column = {
        val exprs = cols.map(cn => new Column(cn).expr)
        new Column(ARC_CombinatorialCountAgg(exprs, cols, nCombination).toAggregateExpression())
    }

    def arc_entropy_agg(cols: String*): Column = {
        val exprs = cols.map(cn => new Column(cn).expr)
        new Column(ARC_EntropyAggExpression(exprs, cols).toAggregateExpression())
    }

    def arc_merge_count_map(counter_map: Column): Column = {
        new Column(ARC_MergeCountMapAgg(counter_map.expr).toAggregateExpression())
    }

}
