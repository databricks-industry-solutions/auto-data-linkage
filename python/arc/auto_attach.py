import importlib.metadata
import importlib.resources

from pyspark import SparkContext
from pyspark.sql import SparkSession


def _auto_attach():
    spark = SparkSession.builder.getOrCreate()

    _auto_attach_enabled = (
            spark.conf.get("spark.databricks.industry.solutions.jar.autoattach", "false") == "true"
    )

    if _auto_attach_enabled:
        sc = SparkContext.getOrCreate()
        _jar_filename = f"arc-{importlib.metadata.version('databricks-arc')}-jar-with-dependencies.jar"
        with importlib.resources.path("arc.lib", _jar_filename) as p:
            _jar_path = p.as_posix()
        JavaURI = getattr(sc._jvm.java.net, "URI")
        JavaJarId = getattr(sc._jvm.com.databricks.libraries, "JavaJarId")
        ManagedLibraryId = getattr(
            sc._jvm.com.databricks.libraries, "ManagedLibraryId"
        )
        ManagedLibraryVersions = getattr(
            sc._jvm.com.databricks.libraries, "ManagedLibraryVersions"
        )
        NoVersion = getattr(ManagedLibraryVersions, "NoVersion$")
        NoVersionModule = getattr(NoVersion, "MODULE$")
        DatabricksILoop = getattr(
            sc._jvm.com.databricks.backend.daemon.driver, "DatabricksILoop"
        )
        converters = sc._jvm.scala.collection.JavaConverters

        JarURI = JavaURI.create("file:" + _jar_path)
        lib = JavaJarId(
            JarURI,
            ManagedLibraryId.defaultOrganization(),
            NoVersionModule.simpleString(),
        )
        libSeq = converters.asScalaBufferConverter((lib,)).asScala().toSeq()

        context = DatabricksILoop.getSharedDriverContextIfExists().get()
        context.registerNewLibraries(libSeq)
        context.attachLibrariesToSpark(libSeq)
