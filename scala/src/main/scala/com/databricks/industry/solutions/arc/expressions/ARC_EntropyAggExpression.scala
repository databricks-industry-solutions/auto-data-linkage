package com.databricks.industry.solutions.arc.expressions

import com.databricks.industry.solutions.arc.expressions.base._
import com.esotericsoftware.kryo.Kryo
import com.esotericsoftware.kryo.io.{Input, Output}
import org.apache.commons.io.output.ByteArrayOutputStream
import org.apache.spark.sql.catalyst.expressions.aggregate.{ImperativeAggregate, TypedImperativeAggregate}
import org.apache.spark.sql.catalyst.expressions.Expression
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.types._

import java.io.ByteArrayInputStream

case class ARC_EntropyAggExpression(
    attributeMap: Map[String, Expression],
    base: Int = 0,
    mutableAggBufferOffset: Int = 0,
    inputAggBufferOffset: Int = 0
) extends TypedImperativeAggregate[EntropyCountAccumulatorMap] {

    override def children: Seq[Expression] = attributeMap.values.toSeq

    override def createAggregationBuffer(): EntropyCountAccumulatorMap =
        EntropyCountAccumulatorMap(
          attributeMap.map(x => (x._1, CountAccumulatorMap()))
        )

    override def update(buffer: EntropyCountAccumulatorMap, input: InternalRow): EntropyCountAccumulatorMap = {
        val result = attributeMap.map { case (cn, expr) =>
            val value = expr.eval(input).toString
            val countMap = buffer.counter(cn)
            (cn, countMap ++ value)

        }
        EntropyCountAccumulatorMap(result)
    }

    override def merge(
        buffer: EntropyCountAccumulatorMap,
        input: EntropyCountAccumulatorMap
    ): EntropyCountAccumulatorMap = {
        val keys = buffer.counter.keySet ++ input.counter.keySet
        EntropyCountAccumulatorMap(
          keys.map { k => (k, buffer.counter(k) merge input.counter(k)) }.toMap
        )
    }

    def logDivisor(countMap: CountAccumulatorMap): Double = {
        if (base == 0) {
            val total = countMap.counter.size
            if (total < 2 | total == 10) 1.0 else math.log10(total)
        } else {
            math.log10(base)
        }
    }

    override def eval(buffer: EntropyCountAccumulatorMap): Any = {
        val entropy = buffer.counter
            .map { case (cn, countMap) =>
                val total = countMap.counter.values.sum
                val entropy = countMap.counter.map { case (_, count) =>
                    val p = count.toDouble / total
                    -p * math.log10(p) / logDivisor(countMap)
                }.sum
                (cn, entropy)
            }
        Utils.buildMapDouble(entropy)
    }

    override def serialize(buffer: EntropyCountAccumulatorMap): Array[Byte] = {
        val toSerialize = EntropyCountNestedList(buffer)
        val output = new ByteArrayOutputStream()
        val kryoOutput = new Output(output)
        serializer.writeObject(kryoOutput, toSerialize)
        output.toByteArray
    }

    override def deserialize(storageFormat: Array[Byte]): EntropyCountAccumulatorMap = {
        val input = new ByteArrayInputStream(storageFormat)
        val kryoInput = new Input(input)
        val readObject = serializer.readObject(kryoInput, classOf[EntropyCountNestedList])
        readObject.toMap
    }

    override def withNewMutableAggBufferOffset(newMutableAggBufferOffset: Int): ImperativeAggregate =
        copy(mutableAggBufferOffset = newMutableAggBufferOffset)

    override def withNewInputAggBufferOffset(newInputAggBufferOffset: Int): ImperativeAggregate =
        copy(inputAggBufferOffset = newInputAggBufferOffset)

    override def nullable: Boolean = false

    override def dataType: DataType = MapType(StringType, DoubleType)

    override protected def withNewChildrenInternal(newChildren: IndexedSeq[Expression]): Expression = {
        val newAttributeMap = attributeMap.keys.zip(newChildren).toMap
        copy(attributeMap = newAttributeMap)
    }

    private def serializer: Kryo = {
        val kryo = new Kryo()
        kryo.register(classOf[(Any, Any)], new com.twitter.chill.Tuple2Serializer)
        kryo.register(classOf[Array[(String, Long)]])
        kryo.register(classOf[Array[(String, Array[(String, Long)])]])
        kryo.register(classOf[EntropyCountNestedList])
        kryo
    }

}
