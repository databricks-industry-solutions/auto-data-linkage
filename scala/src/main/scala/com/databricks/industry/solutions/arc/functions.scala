package com.databricks.industry.solutions.arc

import com.databricks.industry.solutions.arc.expressions._
import org.apache.spark.sql.{Column, DataFrame}
import org.apache.spark.sql.functions.{array, lit}

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
        val exprs = cols.map(cn => cn -> new Column(cn).expr).toMap
        new Column(ARC_EntropyAggExpression(exprs).toAggregateExpression())
    }

    def arc_entropy_agg(cols: java.util.ArrayList[String]): Column = {
        arc_entropy_agg(cols.asScala: _*)
    }

    def arc_merge_count_map(counter_map: Column): Column = {
        new Column(ARC_MergeCountMapAgg(counter_map.expr).toAggregateExpression())
    }

    def arc_generate_combinations(n: Int, elements: Column): Column = {
        new Column(ARC_GenerateCombinations(lit(n).expr, elements.expr))
    }

    def arc_generate_combinations(n: Int, elements: String*): Column = {
        arc_generate_combinations(n, array(elements.map(el => lit(el)): _*))
    }

    def arc_generate_combinations(n: Int, elements: java.util.ArrayList[String]): Column = {
        arc_generate_combinations(n, elements.asScala: _*)
    }

    def arc_generate_partial_combinations(n: Int, partials: Column, elements: Column): Column = {
        new Column(ARC_GeneratePartialCombinations(lit(n).expr, partials.expr, elements.expr))
    }

    def arc_combinations(n: Int, elements: Column): Column = {
        new Column(ARC_Combinations(lit(n).expr, elements.expr))
    }

    def arc_combinations(n: Int, elements: String*): Column = {
        arc_combinations(n, array(elements.map(el => lit(el)): _*))
    }

    def arc_combinations(n: Int, elements: java.util.ArrayList[String]): Column = {
        arc_combinations(n, elements.asScala: _*)
    }

    def arc_to_splink_rule(rule_combination: Column): Column = {
        new Column(ARC_ToSplinkRule(rule_combination.expr))
    }

    def arc_estimate_squared_count_or(rule_combination: Column, rule_square_count_map: Map[String, (Long, Long)]): Column = {
        new Column(ARC_EstimateSquaredCountOR(rule_combination.expr, rule_square_count_map))
    }

    def arc_generate_blocking_rules(df: DataFrame, n: Int, k: Int, attributes: String*): DataFrame = {
        ARC.generateBlockingRules(df, n, k, attributes)
    }

    def arc_generate_blocking_rules(df: DataFrame, n: Int, k: Int, attributes: java.util.ArrayList[String]): DataFrame = {
        ARC.generateBlockingRules(df, n, k, attributes.asScala)
    }

}
