package com.databricks.industry.solutions.arc.expressions.base

case class EntropyCountNestedList(counters: Array[(String, Array[(String, Long)])]) {

    def this() = this(Array.empty)

    def toMap: EntropyCountAccumulatorMap =
        EntropyCountAccumulatorMap(counters.toMap.mapValues(values => CountAccumulatorMap(values.toMap)))

}

object EntropyCountNestedList {

    def apply(counters: EntropyCountAccumulatorMap): EntropyCountNestedList = {
        EntropyCountNestedList(counters.counter.mapValues(_.counter.toArray).toArray)
    }

}
