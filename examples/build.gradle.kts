import org.gradle.plugins.javascript.envjs.http.simple.SimpleHttpFileServerFactory
import org.jetbrains.kotlin.gradle.targets.js.webpack.KotlinWebpackOutput.Target.UMD

plugins{
    kotlin("multiplatform")
}

repositories {
    mavenCentral()
}

group "io.github.kgpu"
version "0.1.0"

kotlin {
    jvm(){
        withJava()
        tasks{
            register<Jar>("jvmFatJar") {
                dependsOn("jvmJar")

                manifest {
                    attributes["Main-Class"] = "DesktopExampleKt"
                }
                archiveBaseName.set("${project.name}-fat")
                from(configurations.getByName("runtimeClasspath").map { if (it.isDirectory) it else zipTree(it) },
                    compilations.getByName("main").output.classesDirs,
                    compilations.getByName("main").output.resourcesDir
                )
            }
        }
    }
    js().browser()

    sourceSets {
        val commonMain by getting {
            dependencies{
                implementation(project(":"))
                implementation(kotlin("stdlib-common"))
            }
        }
        val jvmMain by getting {
            dependencies {
                implementation(kotlin("stdlib-jdk8"))
            }
        }
        val jsMain by getting{
            dependencies {
                implementation(kotlin("stdlib-js"))
            }
        }
    }
}

tasks {
    register("buildWebExample") {
        dependsOn("jsBrowserDistribution")
        dependsOn("jsBrowserWebpack")
    }

    register("startWebServer"){
        val port = 8080;
        val path = "$buildDir/distributions"

        doLast {
            val server = SimpleHttpFileServerFactory().start(File(path), port)

            println("Server started in directory " + server.getContentRoot())
            println("Link: http://localhost:" + server.getPort() + "/index.html\n\n")
        }
    }

    register("runTriangleExample", Exec::class){
        dependsOn("jvmFatJar")

        workingDir("$projectDir")
        commandLine("java", "-jar", "$buildDir/libs/examples-fat.jar", "-triangle")
    }

    register("runCubeExample", Exec::class){
        dependsOn("jvmFatJar")

        workingDir("$projectDir")
        commandLine("java", "-jar", "$buildDir/libs/examples-fat.jar", "-cube")
    }

    register("runTextureExample", Exec::class){
        dependsOn("jvmFatJar")

        workingDir("$projectDir")
        commandLine("java", "-jar", "$buildDir/libs/examples-fat.jar", "-texture")
    }

    register("runEarthExample", Exec::class){
        dependsOn("jvmFatJar")

        workingDir("$projectDir")
        commandLine("java", "-jar", "$buildDir/libs/examples-fat.jar", "-earth")
    }
}