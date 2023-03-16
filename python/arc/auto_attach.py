from pyspark import SparkContext


def _auto_attach():
    sc = SparkContext.getOrCreate()
    _jar_filename = f"mosaic-{importlib.metadata.version('databricks-arc')}-jar-with-dependencies.jar"
    JavaURI = getattr(self.sc._jvm.java.net, "URI")
    JavaJarId = getattr(self.sc._jvm.com.databricks.libraries, "JavaJarId")
    ManagedLibraryId = getattr(
        self.sc._jvm.com.databricks.libraries, "ManagedLibraryId"
    )
    ManagedLibraryVersions = getattr(
        self.sc._jvm.com.databricks.libraries, "ManagedLibraryVersions"
    )
    NoVersion = getattr(ManagedLibraryVersions, "NoVersion$")
    NoVersionModule = getattr(NoVersion, "MODULE$")
    DatabricksILoop = getattr(
        sc._jvm.com.databricks.backend.daemon.driver, "DatabricksILoop"
    )
    converters = self.sc._jvm.scala.collection.JavaConverters

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
