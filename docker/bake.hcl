// ============================================================
// docker buildx bake definition for TheAgency's modular image stack.
//
// Each target maps to one Dockerfile.<name> and is tagged theagency:<name>.
// A target's `contexts` entry lets its Dockerfile say a plain
// `FROM theagency:<parent>` and have bake resolve that to the PARENT
// TARGET's build output directly -- no registry push/pull needed, and the
// whole DAG builds in one command with correct ordering and shared cache:
//
//   base_cuda -> ros2_humble -> perception -> robotican
//                                          \-> detector
//                                          \-> xtend   (future sibling)
//
// Usage:
//   ./docker/stage_vendor_msgs.sh          # vendor .msg sources, once
//   docker buildx bake -f docker/bake.hcl robotican
// ============================================================
variable "REGISTRY" {
  default = "theagency"
}

group "default" {
  targets = ["robotican"]
}

target "base_cuda" {
  context    = "docker"
  dockerfile = "Dockerfile.base_cuda"
  tags       = ["${REGISTRY}:base_cuda"]
}

target "ros2_humble" {
  context    = "docker"
  dockerfile = "Dockerfile.ros2_humble"
  tags       = ["${REGISTRY}:ros2_humble"]
  contexts   = {
    "theagency:base_cuda" = "target:base_cuda"
  }
}

target "perception" {
  context    = "docker"
  dockerfile = "Dockerfile.perception"
  tags       = ["${REGISTRY}:perception"]
  contexts   = {
    "theagency:ros2_humble" = "target:ros2_humble"
  }
}

target "robotican" {
  context    = "docker"
  dockerfile = "Dockerfile.robotican"
  tags       = ["${REGISTRY}:robotican"]
  contexts   = {
    "theagency:perception" = "target:perception"
  }
  args = {
    USERNAME = "user1"
    USER_UID = "1000"
    USER_GID = "1000"
  }
}

target "detector" {
  context    = "docker"
  dockerfile = "Dockerfile.detector"
  tags       = ["${REGISTRY}:detector"]
  contexts   = {
    "theagency:perception" = "target:perception"
  }
  args = {
    USERNAME = "user1"
    USER_UID = "1000"
    USER_GID = "1000"
  }
}

// Future sibling, once the XTEND stack is built -- same perception/ros2_humble/
// base_cuda parents, its own (probably much smaller) leaf layer:
//
// target "xtend" {
//   context    = "docker"
//   dockerfile = "Dockerfile.xtend"
//   tags       = ["${REGISTRY}:xtend"]
//   contexts   = {
//     "theagency:perception" = "target:perception"
//   }
// }
