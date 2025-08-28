from conan import ConanFile
from conan.tools.cmake import CMakeDeps, CMakeToolchain

class TrackieLinkConan(ConanFile):
    name = "trackielink"
    version = "1.0"
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "CMakeToolchain"

    def requirements(self):
        self.requires("opencv/4.5.5")
        self.requires("libcurl/8.6.0")
        self.requires("onnxruntime/1.16.3")
        self.requires("spdlog/1.12.0")
        self.requires("gtest/1.14.0")

    def layout(self):
        self.folders.build = "build"
        # The generator files will be placed in the 'build' folder.
        # This is consistent with the --output-folder=build argument
        # used in the build scripts.
        self.folders.generators = "build"
