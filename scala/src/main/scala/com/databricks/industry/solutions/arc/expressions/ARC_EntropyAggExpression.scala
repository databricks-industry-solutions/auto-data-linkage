package com.databricks.industry.solutions.arc.expressions

import com.databricks.industry.solutions.arc.expressions.base.{CountAccumulatorMap, Utils}
import org.apache.commons.lang3.SerializationUtils
import org.apache.spark.sql.catalyst.expressions.aggregate.{ImperativeAggregate, TypedImperativeAggregate}
import org.apache.spark.sql.catalyst.expressions.Expression
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.types._

case class ARC_EntropyAggExpression(
    attributeExprs: Seq[Expression],
    attributeNames: Seq[String],
    mutableAggBufferOffset: Int = 0,
    inputAggBufferOffset: Int = 0
) extends TypedImperativeAggregate[Map[String, CountAccumulatorMap]] {

    private val attributeMap = attributeNames.zip(attributeExprs).toMap

    override def children: Seq[Expression] = attributeMap.values.toSeq

    override def createAggregationBuffer(): Map[String, CountAccumulatorMap] =
        attributeMap.map(x => (x._1, CountAccumulatorMap())) + ("total" -> CountAccumulatorMap())

    override def update(buffer: Map[String, CountAccumulatorMap], input: InternalRow): Map[String, CountAccumulatorMap] = {
        val result = attributeMap.map { case (cn, expr) =>
            val value = expr.eval(input).toString
            val countMap = buffer(cn)
            (cn, countMap ++ value)

        } + ("total" -> (buffer("total") ++ "total"))
        result
    }

    override def merge(
        buffer: Map[String, CountAccumulatorMap],
        input: Map[String, CountAccumulatorMap]
    ): Map[String, CountAccumulatorMap] = {
        val keys = buffer.keySet ++ input.keySet
        keys.map { k => (k, buffer(k) merge input(k)) }.toMap
    }

    def logDivisor(countMap: CountAccumulatorMap): Double = {
        val total = countMap.counter.size
        if (total <= 2) 1.0 else math.log(total)
    }

    override def eval(buffer: Map[String, CountAccumulatorMap]): Any = {
        val total = buffer("total").counter("total")
        val entropy = buffer
            .map { case (cn, countMap) =>
                val entropy = countMap.counter.map { case (_, count) =>
                    val p = count.toDouble / total
                    -p * math.log(p) / logDivisor(countMap)
                }.sum
                (cn, entropy)
            }
            .filterNot(_._1 == "total")
        Utils.buildMapDouble(entropy)
    }

    override def serialize(buffer: Map[String, CountAccumulatorMap]): Array[Byte] =
        SerializationUtils.serialize(buffer.asInstanceOf[Serializable])

    override def deserialize(storageFormat: Array[Byte]): Map[String, CountAccumulatorMap] =
        SerializationUtils.deserialize(storageFormat).asInstanceOf[Map[String, CountAccumulatorMap]]

    override def withNewMutableAggBufferOffset(newMutableAggBufferOffset: Int): ImperativeAggregate =
        copy(mutableAggBufferOffset = newMutableAggBufferOffset)

    override def withNewInputAggBufferOffset(newInputAggBufferOffset: Int): ImperativeAggregate =
        copy(inputAggBufferOffset = newInputAggBufferOffset)

    override def nullable: Boolean = false

    override def dataType: DataType = MapType(StringType, DoubleType)

    override protected def withNewChildrenInternal(newChildren: IndexedSeq[Expression]): Expression = copy(attributeExprs = newChildren)

}
