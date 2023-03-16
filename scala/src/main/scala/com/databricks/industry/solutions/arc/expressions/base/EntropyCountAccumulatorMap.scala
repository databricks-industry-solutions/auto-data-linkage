package com.databricks.industry.solutions.arc.expressions.base

case class EntropyCountAccumulatorMap(map: Map[String, Map[String, Long]]) extends Serializable
