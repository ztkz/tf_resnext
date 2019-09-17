from absl import flags

from tf_resnext.models import ResidualType

FLAGS = flags.FLAGS

flags.DEFINE_list(
    "training_schedule",
    ["150", "75", "75", "75"],
    "The numbers of epochs with decaying the learning rate in-between.",
)
flags.DEFINE_integer("cardinality", 16, "The cardinality of grouped convolutions.")
flags.DEFINE_integer("depth", 29, "The depth of the network.")
flags.DEFINE_integer("base_width", 64, "The base width of the network.")
flags.DEFINE_enum("residual_type", ResidualType.ORIGINAL.name,
                  [value.name for value in ResidualType], "The type of the residual connection")
flags.DEFINE_float("learning_rate", 0.05, "The initial learning rate.")
flags.DEFINE_float("learning_rate_decay_factor", 0.1, "The decay factor of the learning rate.")
flags.DEFINE_float("weight_decay", 5e-4 * 0.5, "The L2 weight decay.")
flags.DEFINE_integer("batch_size", 64, "The batch size.")
flags.DEFINE_float("momentum", 0.9, "The momentum.")
flags.DEFINE_bool("use_nesterov", True, "Whether to use Nesterov Momentum.")
flags.DEFINE_float("per_process_gpu_memory_fraction", 0.85, "GPU memory fraction to use.")
