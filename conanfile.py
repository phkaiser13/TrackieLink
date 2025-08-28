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
        # We will use a build folder for the build outputs
        self.folders.build = "build"
        # The generated files (conan_toolchain.cmake, etc.) will be in the build/generators folder
        self.generators_folder = "generators"
