package com.databricks.industry.solutions.arc.expressions.base

class EntropyCountAccumulatorMap(_counter: Map[String, CountAccumulatorMap]) extends Serializable {

    def counter: Map[String, CountAccumulatorMap] = this._counter

    def this() = this(Map.empty)

    def preSerialize: CountAccumulatorMap =
        CountAccumulatorMap(
          counter.map { case (k, v) => v.counter.keys.map(k1 => s"${k}__$k1").zip(v.counter.values) }.reduce(_ ++ _).toMap
        )

}

object EntropyCountAccumulatorMap {

    def apply(map: Map[String, CountAccumulatorMap]): EntropyCountAccumulatorMap = new EntropyCountAccumulatorMap(map)

    def postDeSerialize(inCounter: CountAccumulatorMap): EntropyCountAccumulatorMap = {
        val result = inCounter.counter.groupBy(_._1.split("____").head)
        EntropyCountAccumulatorMap(
          result.mapValues(m => CountAccumulatorMap(m.map { case (k, v) => k.split("____").last -> v }))
        )
    }

}
