# import data.basketball.proto.possession_pb2 as possession
import os, sys, numpy, datetime, itertools, copy, math
import traceback
from utils.u_logging.logging import *

numpy.set_printoptions(threshold=numpy.nan)

from shutil import copyfile

import struct
# import threading

from multiprocessing import Pool
import Queue

import tensorflow as tf
import tensorflow.python.platform

from rl.bball.constants import *
from rl.bball.models.factored_lstm_constants import *
import rl.bball.imitation.keras.constants as constants

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('parse_simple', False, "Parse full or simple version of tracking data.")
flags.DEFINE_boolean('test', False, "Parse full or simple version of tracking data.")
flags.DEFINE_boolean('verbose', False, "LOG galore.")
flags.DEFINE_boolean('visualize_tracks', False, "Write SVG files for player tracks.")
flags.DEFINE_boolean('write_tracks', False, "Process tracks and write to file.")

flags.DEFINE_boolean('use_multiple_start_times', False, "Extract all possible subsequences of length FLAGS.encoder_input_length, with some M steps skip, of the full track?.")

flags.DEFINE_boolean('use_sparse', False, "Process tracks and write to file.")

flags.DEFINE_boolean('use_cell_coordinates', True, ".")
flags.DEFINE_boolean('use_float32_coordinates', False, ".")

flags.DEFINE_boolean('do_stats', False, ".")

flags.DEFINE_boolean('use_track_segmentation_for_macro', False, ".")

flags.DEFINE_integer('num_threads', 20, ".")

flags.DEFINE_integer('encoder_input_length', 100, "Length of input sentence.")

flags.DEFINE_integer('subsample_steps', 1, "Write only every # timesteps. Note that the final sequence length is FLAGS.encoder_input_length / FLAGS.subsample_steps.")

flags.DEFINE_integer('num_frames_ahead_for_microlabels', 25, "How many lookahead labels do we use for the micro-action labels.")
flags.DEFINE_integer('num_frames_ahead_for_macrogoal', 25, "How far ahead do we extrapolate to infer what the macro-goal is? (25 frames = 1 second).")
flags.DEFINE_integer('num_macro_labels', 1, "How many macro goals do we use. If > 1, we use regular spaced samples up to num_frames_ahead_for_macrogoal")

flags.DEFINE_integer('num_possessions', -1, "How many possessions to process. If < 0, just do all.")

flags.DEFINE_string('www_svg_dir', "/home/stzheng/projects/www/tracks/img/", "Where the SVG of the tracks should be written.")
flags.DEFINE_string('simple_data_dir', "/cs/ml/datasets/bball_tracking/basketball_data/data", "Where the SVG of the tracks should be written.")
flags.DEFINE_string('full_data_dir', "/tmp/stephan/data/basketball/possession_data", "Where the SVG of the tracks should be written.")
flags.DEFINE_string('output_dir', "/tmp/stephan/data/basketball/parsed/public", "Where the SVG of the tracks should be written.")


# There are 2 versions of the basketball data: simple and full.
SIMPLE_DATA = "/cs/ml/datasets/bball_tracking/basketball_data/data"
FULL_DATA = "/cs/ml/datasets/bball/v2/possession_data/"

id_queue = Queue.Queue()



def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

SCALE = 1.

v_upscale_factor = 1
v_upscale_factor_inv = 1 / 1. # have to use inverse

DOMAIN_SCALE = 8
DOMAIN_HEIGHT = 50
DOMAIN_WIDTH = 45
MICRO_DOMAIN_HEIGHT = 11
MICRO_DOMAIN_WIDTH = 11
MACRO_DOMAIN_SCALE = 9
MACRO_DOMAIN_WIDTH = 8
MG_GRID_X_MAX = 17
MG_GRID_Y_MAX = 17

class Parser:

  def __init__(self, data_dir="", output_dir="", parse_simple=False):

    self.data_dir = data_dir
    self.output_dir = output_dir

    self.possession_counter = 0

    self.size_x  = 1.0
    self.size_y  = 1.0
    self.size_z  = 1.0

    # Raw coordinates ranges
    self.phys_range_x = 50.
    self.phys_range_y = 45.
    self.phys_range_z = 10.

    # Going from raw coordinates to discretization scale
    self.scale_domain_by = constants.domain_scale

    self.num_domain_cells_x = constants.domain_height
    self.num_domain_cells_y = constants.domain_width

    # Velocity

    self.V_MIN = 0.
    self.V_MAX = 0.

    # We use these numbers that we see from data.
    self.num_vx_cells = constants.micro_domain_height
    self.num_vy_cells = constants.micro_domain_width

    # When using float32 coordinates throughout
    self.vx_min = -1.0
    self.vy_min = self.vx_min
    self.vx_max = 1.0
    self.vy_max = self.vx_max
    self.vx_cell_width = (self.vx_max - self.vx_min) / float(self.num_vx_cells)
    self.vy_cell_width = (self.vy_max - self.vy_min) / float(self.num_vy_cells)

    # When using cell coordinates throughout
    self.vx_min_cell = -(self.num_vx_cells - 1) / 2
    self.vy_min_cell = self.vx_min_cell
    self.vx_max_cell = -self.vx_min_cell
    self.vy_max_cell = self.vx_max_cell

    self.vx_center_cell = (self.num_vx_cells - 1) / 2
    self.vy_center_cell = (self.num_vy_cells - 1) / 2

    self.V_DISCARD_THRESHOLD = 2 * self.vy_max_cell # if the cell v is too big, probably data corruption

    # We count stats using a different grid than the parsing routine.
    self.vel_stats_dim_x = self.num_vx_cells
    self.vel_stats_dim_y = self.vel_stats_dim_x
    self.vel_stats = numpy.zeros((self.vel_stats_dim_x, self.vel_stats_dim_y), dtype=numpy.int32)
    self.vel_stats_min = self.vx_min_cell
    self.vel_stats_max = self.vy_min_cell
    self.vel_cell_resolution = (self.vel_stats_max - self.vel_stats_min) / self.vel_stats_dim_x

  def parse_full_data_as_fixlen_raw(self, team_dir):

    try:

      self.thread_id = id_queue.get()
      LOG("Thread", self.thread_id, "Going to process dir:", os.path.join(os.getcwd(), team_dir))

      try:
        assert "Team" in team_dir
        LOG("Thread", self.thread_id, team_dir, "- name pattern ok")
      except:
        LOG("Thread", self.thread_id, team_dir, "- folder name does not conform to pattern")

      old_dir = os.getcwd()

      if "Team" in team_dir:
        os.chdir(team_dir)
      else:
        LOG("Skipping file / dir:", team_dir)
        return

      dir_name = team_dir.split("_")
      team_id = int(dir_name[-1])
      files_in_teamdir = os.listdir(os.getcwd())
      files_in_teamdir.sort()

      # Loop over possessions in the team directory.
      assert len(files_in_teamdir) % 4 == 0
      for quadruple_idx in xrange(len(files_in_teamdir) / 4):

        quadruple = files_in_teamdir[4*quadruple_idx:4*quadruple_idx+4]

        if FLAGS.test:
          if quadruple_idx == 10 and team_id == 24:
            LOG_WITH_SPACE("Thread", self.thread_id, "Processing from team", team_id)
            success = self.process_possession(dir_name, quadruple)

          continue
        else:
          success = self.process_possession(dir_name, quadruple)

        if success:
          self.possession_counter += 1

        if self.possession_counter % 1000 == 0:
          LOG("Thread", self.thread_id, "Loaded", self.possession_counter, "possessions")

          LOG("Thread", self.thread_id, "vel_stats", self.vel_stats)


        if FLAGS.num_possessions > 0 and self.possession_counter >= FLAGS.num_possessions:
          LOG("Thread", self.thread_id, "reached", FLAGS.num_possessions, "--> quitting")
          break

      # Parsed all possessions, go to parent dir
      os.chdir(old_dir)

    except Exception as e:
      print('Caught exception in worker thread', self.thread_id)

      # This prints the type, value, and stack trace of the
      # current exception being handled.
      traceback.print_exc()

      print()
      raise e

  def process_possession(self, name, quadruple):

    file_meta       = quadruple[0]
    file_optical    = quadruple[1]
    file_pbp        = quadruple[2]
    file_timeStamps = quadruple[3]

    if FLAGS.verbose:
      LOG("Processing", quadruple)

    assert file_meta.split("_")[1] == file_optical.split("_")[1] == file_pbp.split("_")[1] == file_timeStamps.split("_")[1]
    try:
      assert "meta" in file_meta and "optical" in file_optical and "pbp" in file_pbp and "timeStamps" in file_timeStamps
    except:
      LOG("Files do not conform to pattern.")

    team_id       = int(name[1])
    possession_id = int(file_meta.split("_")[1])

    # Process meta-info
    with open(file_meta, "r") as _file_meta:
      lines           = [line for line in _file_meta]

      outcome         = int(lines[0])
      team_id_1       = int(lines[1])
      team_id_2       = int(lines[2])
      teamA_direction = int(lines[3])
      possession      = int(lines[4])
      frame_start     = int(lines[5])
      frame_end       = int(lines[6])
      pbp_start       = int(lines[7])
      pbp_end         = int(lines[8])
      duration        = float(lines[9])

      if FLAGS.verbose:
        LOG_WITH_SPACE("teamA_direction", teamA_direction)

      if possession == 1:
        offense_team = team_id_1
        defense_team = team_id_2
      else:
        offense_team = team_id_2
        defense_team = team_id_1

      assert offense_team == team_id

      if FLAGS.verbose:
        LOG("outcome", outcome)
        LOG("frame_start", frame_start)
        LOG("frame_end", frame_end)
        LOG("pbp_start", pbp_start)
        LOG("pbp_end", pbp_end)
        LOG("duration", duration)
        LOG("team_id_1", team_id_1)
        LOG("team_id_2", team_id_2)
        LOG("teamA_direction", teamA_direction)
        LOG("possession", possession)
        LOG("offense_team", offense_team)
        LOG("defense_team", defense_team)


    mirror = (possession == 1 and teamA_direction == 2) or (possession == 2 and teamA_direction == 1)
    # if mirror:
      # LOG("Flipping in x:", "possession", possession, "teamA_direction", teamA_direction)


    # Process play-by-play-data
    ballhandler_ids = []
    with open(file_pbp, "r") as _file_pbp:
      pbp_lines = [line.replace("\n", "").split(",") for line in _file_pbp]
      pbp_lines = [[int(line[0]), int(line[1]), float(line[2]), int(line[3])] for line in pbp_lines]

      self.pbp_lines = pbp_lines

      if FLAGS.verbose:
        LOG("pbp:", pbp_lines)

      for pbp_line in pbp_lines:
        unknown_id     = int(pbp_line[0]) # todo(stz): figure out what this is.
        event_id       = int(pbp_line[1])
        timestamp      = float(pbp_line[2])
        ballhandler_id = int(pbp_line[3])

        ballhandler_ids.append(ballhandler_id)

      if FLAGS.verbose:
        LOG("# todo(stz): figure out what this is. unknown_id", unknown_id)
        LOG("event_id", event_id)
        LOG("timestamp", timestamp)
        LOG("ballhandler_id", ballhandler_id)

      # LOG("ballhandler_ids", ballhandler_ids)

    self.tracks_xyt         = {}
    self.tracks_xyt_upscaled= {}
    self.track_history_xyzt = {}
    self.tracks_vel_xyt     = {}
    self.ball_xyzt          = {}
    self.labels             = {}
    self.segments           = {}

    # Process the raw optical data
    with open(file_optical, "r") as _file_optical, open(file_timeStamps, "r") as _file_timeStamps:

      pbp_idx = 0 # each t, we check if an event occurred.

      for frame_id, (line_optical, line_timeStamps) in enumerate(itertools.izip(_file_optical, _file_timeStamps)):
        timestamp = float(line_timeStamps)

        if pbp_idx < len(self.pbp_lines):
          pbp_timestamp = self.pbp_lines[pbp_idx][2]

          if timestamp >= pbp_timestamp:

            ballhandler_id = self.pbp_lines[pbp_idx][3]
            pbp_idx += 1

        is_ballhandler = True
        success = self.process_frame(offense_team, defense_team, possession_id, frame_id, line_optical, line_timeStamps, mirror=mirror, ballhandler_id=ballhandler_id, is_ballhandler=is_ballhandler)

        if not success:
          LOG("Possession not successfully processed -- probably v went OOB. Not using this possession.")
          return success

    num_frames = max([len(self.tracks_xyt[key]) for key in self.tracks_xyt.keys()])
    if FLAGS.verbose:
      LOG_WITH_SPACE("num_frames", num_frames)


    # Once we have the raw (x,y,t)-values, we can compute macro-goals.
    # Writes to self.labels
    key = self.get_full_key(self.track_history_xyzt, ballhandler_id)

    if FLAGS.use_track_segmentation_for_macro:
      self.segments[key] = self.get_track_segments(self.track_history_xyzt, ballhandler_id)
      # LOG("segms:", key, self.segments[key])
      if self.segments[key]:
        self.labels[key] = self.get_macrogoals_from_segments(self.tracks_xyt, key)
      else:
        self.labels[key] = [-1]
    else:
      self.labels[key] = self.get_possession_macrogoals(self.track_history_xyzt, ballhandler_id)

    # Write tracks
    if FLAGS.write_tracks and all([len(i) > 0 for i in self.tracks_vel_xyt.values()]) and len(self.labels[key]) > 0:
      if self.labels[key][0] == -1:
        LOG("No macro-goals were constructed, so NOT writing this possession to disk... :(")
      else:
        self.write_tracks(offense_team, defense_team, possession_id, ballhandler_id)

    # Write to visualization
    if FLAGS.visualize_tracks:
      for k,track_xyt in self.tracks_xyt.iteritems():
        self.xyt_to_svg_as_csv(self.track_xyt, self.output_dir + "csv/" + "_".join( [str(kk) for kk in k] ) + ".csv")
      self.xyzt_to_svg_as_csv(self.ball_xyzt[(possession_id)], self.output_dir + "csv/" + "ball-%i.csv" % possession_id)

    LOG("Parsed ok:", "poss-id", possession_id, "team-id", team_id)
    return True

  def process_frame(self, offense_team, defense_team, possession_id, frame_id, line_optical, line_timeStamps, mirror=False, ballhandler_id=0, is_ballhandler=False):

    timestamp = float(line_timeStamps)
    items_optical = line_optical.split(",")

    ball_loc    = [float(i) * SCALE for i in items_optical[:3]]
    some_ints   = [int(i) for i in items_optical[3:8]]
    offense_loc = [float(i) * SCALE for i in items_optical[8:18]]
    defense_loc = [float(i) * SCALE for i in items_optical[18:28]]
    offense_ids = [int(i) for i in items_optical[28:33]]
    defense_ids = [int(i) for i in items_optical[33:38]]


    if ballhandler_id not in offense_ids:
      temp = offense_team
      offense_team = defense_team
      defense_team = temp

      # LOG("flipping off / def: bh not in offense...", ballhandler_id, offense_ids, "def:", defense_ids)

    if FLAGS.verbose:
      LOG("ball_loc:", ball_loc)
      LOG("some_ints:", some_ints)
      LOG("offense_ids:", offense_ids)
      LOG("offense_loc:", offense_loc)
      LOG("defense_ids:", defense_ids)
      LOG("defense_loc:", defense_loc)


    # Process players
    for idx, player in enumerate(offense_ids + defense_ids):
      if player in offense_ids:
        key = (possession_id, offense_team, player)
        coords = offense_loc
      else:
        key = (possession_id, defense_team, player)
        coords = defense_loc

      if key not in self.tracks_xyt.keys():
        self.tracks_xyt[key] = []
        self.tracks_xyt_upscaled[key] = []
      if key not in self.track_history_xyzt.keys():
        self.track_history_xyzt[key] = []

      idx_modded = idx % len(offense_ids)

      # Bound and flip raw coordinates so they lie in the left half court.
      xyzt = self.normalize_coords(x=coords[2*idx_modded], y=coords[2*idx_modded+1], t=timestamp, mirror=mirror)
      self.scale_by(self.scale_domain_by, xyzt)

      # if key[2] == ballhandler_id:
      #   LOG("xyzt bh  :", xyzt)

      self.track_history_xyzt[key].append(xyzt)
      # LOG("xyzt rescaled:", xyzt)

      if FLAGS.use_sparse:
        sparse_index = self.xy_to_serial_col_maj(x=xyzt[0], y=xyzt[1], x_range=self.num_domain_cells_x, y_range=self.num_domain_cells_y, verbose=False)
        self.tracks_xyt[key].append(sparse_index)

        sparse_index = self.xy_to_serial_col_maj(x=xyzt[0], y=xyzt[1], x_range=self.num_domain_cells_x, y_range=self.num_domain_cells_y, x_scale=v_upscale_factor_inv, y_scale=v_upscale_factor_inv, verbose=False)
        self.tracks_xyt_upscaled[key].append(sparse_index)

      if FLAGS.use_cell_coordinates:
        success = self.process_frame_velocity_cell(key)
        if not success:
          return False

      elif FLAGS.use_float32_coordinates:
        self.process_frame_velocity_float32(key)

    # _sample_key = self.track_history_xyzt.keys()[0]
    # LOG("xy:", self.track_history_xyzt[_sample_key])
    # LOG("vxvy:", self.tracks_vel_xyt[_sample_key])

    # Process ball
    key = (possession_id)
    if key not in self.ball_xyzt.keys():
      self.ball_xyzt[key] = []

    xyzt = self.normalize_coords(x=ball_loc[0], y=ball_loc[1], z=ball_loc[2], t=timestamp, mirror=mirror)

    self.scale_by(self.scale_domain_by, xyzt)

    # LOG("xyzt ball:", xyzt)

    if FLAGS.use_sparse:
      ball_sparse_index = self.xy_to_serial_col_maj(x=xyzt[0], y=xyzt[1], x_range=self.num_domain_cells_x, y_range=self.num_domain_cells_y)
      self.ball_xyzt[key].append( ball_sparse_index )

    return True

  def process_frame_velocity_cell(self, key):
    """Takes xyz in cell coordinates and computes cell velocity.
    If the velocity exceed a certian threshold, we discard the track. This is usually due to data corruption of sorts."""
    if key not in self.tracks_vel_xyt.keys():
      self.tracks_vel_xyt[key] = []

    if len(self.tracks_xyt[key]) > 1: # we've seen at least 1 frame, so we can compute the velocity

      if v_upscale_factor != 1:
        serial_index     = self.tracks_xyt_upscaled[key][-1]
        cell_xy_index    = self.serial_to_xy_col_maj(serial_index, rows=v_upscale_factor * self.num_domain_cells_x, cols=v_upscale_factor * self.num_domain_cells_y)

        serial_index_old = self.tracks_xyt_upscaled[key][-2]
        cell_xy_index_old= self.serial_to_xy_col_maj(serial_index_old, rows=v_upscale_factor * self.num_domain_cells_x, cols=v_upscale_factor * self.num_domain_cells_y)

      else:
        serial_index     = self.tracks_xyt[key][-1]
        cell_xy_index    = self.serial_to_xy_col_maj(serial_index, rows=self.num_domain_cells_x, cols=self.num_domain_cells_y)

        serial_index_old = self.tracks_xyt[key][-2]
        cell_xy_index_old= self.serial_to_xy_col_maj(serial_index_old, rows=self.num_domain_cells_x, cols=self.num_domain_cells_y)



      v_xyzt = cell_xy_index[0] - cell_xy_index_old[0], cell_xy_index[1] - cell_xy_index_old[1], 0, 1


      if abs(int(v_xyzt[0])) > self.V_DISCARD_THRESHOLD or abs(int(v_xyzt[1])) > self.V_DISCARD_THRESHOLD:
        LOG("v went OOB:", v_xyzt, self.V_DISCARD_THRESHOLD)
        return False

      # Now map this velocity to the vel grid, where the center corresponds to v = (0, 0)
      vx = int(v_xyzt[0]) - self.vx_min_cell
      vy = int(v_xyzt[1]) - self.vy_min_cell

      if FLAGS.do_stats:
        self.do_stats(v_xyzt)

      if FLAGS.use_sparse:
        sparse_index = self.xy_to_serial_col_maj_cell(x=vx, y=vy, rows=self.num_vx_cells, cols=self.num_vy_cells)

        if sparse_index > self.num_vx_cells * self.num_vy_cells - 1 or sparse_index < 0:
          LOG("ohoh", serial_index, serial_index_old, "xy", cell_xy_index, "<--", cell_xy_index_old, "raw v_xyzt", v_xyzt, "vel grid coords", vx, vy, "vel_grid:", sparse_index)

        sparse_index = self.bound(sparse_index, 0, self.num_vx_cells * self.num_vy_cells - 1)

        self.tracks_vel_xyt[key].append( sparse_index )

      return True # good cell
    else:
      if FLAGS.use_sparse:
        assert VELOCITY_XY_GRID_XCELLS == VELOCITY_XY_GRID_YCELLS
        k = (VELOCITY_XY_GRID_XCELLS - 1) / 2
        center_cell = 2 * k * k + 2 * k + 1
        self.tracks_vel_xyt[key].append( center_cell )
        return True

  def process_frame_velocity_float32(self, key):
    """Takes xyz in cell coordinates and computes cell velocity."""
    if key not in self.tracks_vel_xyt.keys():
      self.tracks_vel_xyt[key] = []

    if len(self.track_history_xyzt[key]) > 1: # we've seen at least 1 frame, so we can compute the velocity
      xyzt     = self.track_history_xyzt[key][-1]
      xyzt_old = self.track_history_xyzt[key][-2]

      v_xyzt   = xyzt[0] - xyzt_old[0], xyzt[1] - xyzt_old[1], xyzt[2] - xyzt_old[2], xyzt[3] - xyzt_old[3]

      # Do stats
      idx_x = int((v_xyzt[0] - self.vel_stats_min) / self.vel_cell_resolution)
      idx_y = int((v_xyzt[1] - self.vel_stats_min) / self.vel_cell_resolution)

      idx_x = self.bound(idx_x, 0, self.vel_stats_dim_x - 1)
      idx_y = self.bound(idx_y, 0, self.vel_stats_dim_y - 1)

      self.vel_stats[idx_x, idx_y] += 1

      if v_xyzt[0] < self.V_MIN:
        self.V_MIN = v_xyzt[0]
      if v_xyzt[0] > self.V_MAX:
        self.V_MAX = v_xyzt[0]
      if v_xyzt[1] < self.V_MIN:
        self.V_MIN = v_xyzt[1]
      if v_xyzt[1] > self.V_MAX:
        self.V_MAX = v_xyzt[1]


      if FLAGS.use_sparse:
        sparse_index = self.dense_to_sparse_action(vx=v_xyzt[0], vy=v_xyzt[1])

        # LOG("xy", self.track_history_xyzt[key][-2], self.track_history_xyzt[key][-1], "vxvy:", v_xyzt, sparse_index)

        self.tracks_vel_xyt[key].append( sparse_index )
    else:
      if FLAGS.use_sparse:
        assert VELOCITY_XY_GRID_XCELLS == VELOCITY_XY_GRID_YCELLS
        k = (VELOCITY_XY_GRID_XCELLS - 1) / 2
        center_cell = 2 * k * k + 2 * k + 1
        self.tracks_vel_xyt[key].append( center_cell )


  # Process coordinates
  def bound(self, in_, l, u):
    if in_ < l:
      in_ = l
    if in_ > u:
      in_ = u
    return in_

  def scale_by(self, scale, array):
    for i in xrange(len(array)):
      array[i] = scale * array[i]

  def normalize_coords(self, x=0, y=0, z=0, t=0, mirror=False):
    """
    Takes floating point coordinates x,y,z,t and transforms them into format that we want.
    Be careful! Indices are 0-indexed. So x and y values have to be in
    [0, range_x - 1], [0, range_y - 1]. !!!
    """
    if mirror:
      # LOG("Mirror this:", x, y,z, t, "using", REAL_COURT_WIDTH, REAL_COURT_HEIGHT, self.phys_range_x, self.phys_range_y)

      x, y, z, t = min(REAL_COURT_WIDTH - x, self.phys_range_x), min(REAL_COURT_HEIGHT - y, self.phys_range_y), z, t

    x, y, z, t = min(x, self.phys_range_x), min(y, self.phys_range_y), z, t
    x, y, z, t = max(x, 0.), max(y, 0.), z, t

    assert x >= 0. and y >= 0. and x <= self.phys_range_x and y <= self.phys_range_y

    return [x, y, z, t]

  def compute_velocity(self, x1=0, x2=0, y1=0, y2=0, z1=0, z2=0, t=0, mirror=False):
    """Be careful! Indices are 0-indexed. So x and y values have to be in
    [0, range_x - 1], [0, range_y - 1]. !!!
    """
    if mirror:
      # LOG("Mirror this:", x, y,z, t, "using", REAL_COURT_WIDTH, REAL_COURT_HEIGHT, self.phys_range_x, self.phys_range_y)

      x, y, z, t = min(REAL_COURT_WIDTH - x, self.phys_range_x - 1.), min(REAL_COURT_HEIGHT - y, self.phys_range_y - 1.), z, t

    x, y, z, t = min(x, self.phys_range_x - 1.), min(y, self.phys_range_y - 1.), z, t
    x, y, z, t = max(x, 0.), max(y, 0.), z, t

    assert x >= 0. and y >= 0. and x <= self.phys_range_x - 1 and y <= self.phys_range_y - 1

    return [x, y, z, t]


  # cell coords versions
  def serial_to_xy_col_maj(self, index, rows=0, cols=0):
    return int(index / float(cols)), int(index % cols)

  def xy_to_serial_col_maj_cell(self, x=0, y=0, rows=0, cols=0):
    x = self.bound(x, 0, rows)
    y = self.bound(y, 0, cols)
    return int(x) * int(cols) + int(y)


  # float32 versions
  def dense_to_sparse_action(self, vx=0, vy=0, vz=0):
    """map velocity of player to a sparse_index in a grid around the player.
    """
    vx = self.bound(vx, self.vx_min, self.vx_max)
    vy = self.bound(vy, self.vy_min, self.vy_max)

    # If velocity is too small, make sure it lands in the center velocity cell (the player stands still).
    if abs(vx) < (self.vx_max - self.vx_min) / self.num_vx_cells:
      vx = 0.
    if abs(vy) < (self.vy_max - self.vy_min) / self.num_vy_cells:
      vy = 0.

    serial_idx = int(self.xy_to_serial_col_maj(x=(vx - self.vx_min) / self.vx_cell_width,
                                               y=(vy - self.vy_min) / self.vy_cell_width,
                                               x_range= self.num_vx_cells,
                                               y_range= self.num_vy_cells))

    # LOG("get vxvy serial (x):", vx, (vx - self.vx_min), (vx - self.vx_min) / self.vx_cell_width)
    # LOG("get vxvy serial (y):", vy, (vy - self.vy_min), (vy - self.vy_min) / self.vy_cell_width)
    # LOG("gets:", self.num_vx_cells, serial_idx)

    return serial_idx

  def xy_to_serial_col_maj(self, x=0., y=0., x_range=1., y_range=1., x_scale=1., y_scale=1., verbose=False):
    """Computes serial index. Assumes that coords have been normalized, such that
    a unit cell has dimension 1 x 1.

    Arguments:
      x {[type]} -- [description]
      y {[type]} -- [description]
      x_range {[type]} -- number of cells in x direction
      y_range {[type]} -- number of cells in y direction

    Returns:
      [type] -- [description]
    """
    x = self.bound(x, 0, x_range - 1)
    y = self.bound(y, 0, y_range - 1)

    num_domain_cells = x_range  / x_scale * y_range  / y_scale

    if verbose:
      LOG("xy:", x, y, x_scale, y_scale, x_range, y_range, "-->", int(x / float(x_scale)) * int(y_range / float(y_scale)), int(y / y_scale) + int(x / x_scale) * round(y_range / y_scale))

    # LOG(int(x / x_scale), int(y / y_scale), round(x_range / x_scale), int(x / x_scale) + int(y / y_scale) * round(x_range / x_scale))

    return int(min(int(x / float(x_scale)) * round(y_range / float(y_scale)) + int(y / float(y_scale)), int(num_domain_cells) - 1)) #  + len(START_VOCAB)


  # Stats
  def do_stats(self, v_xyzt):
    # Do stats
    idx_x = v_xyzt[0] - self.vx_min_cell
    idx_y = v_xyzt[1] - self.vy_min_cell

    idx_x = self.bound(idx_x, 0, self.num_vx_cells - 1)
    idx_y = self.bound(idx_y, 0, self.num_vy_cells - 1)

    self.vel_stats[idx_x, idx_y] += 1

    if v_xyzt[0] < self.V_MIN:
      self.V_MIN = v_xyzt[0]
    if v_xyzt[0] > self.V_MAX:
      self.V_MAX = v_xyzt[0]
    if v_xyzt[1] < self.V_MIN:
      self.V_MIN = v_xyzt[1]
    if v_xyzt[1] > self.V_MAX:
      self.V_MAX = v_xyzt[1]


  # Sequence modding
  def pad_sequences(self, tracks_dict, desired_seq_length):
    """This method optionally pads sequences with the last value seen in the
    data, up to the desired_seq_length. This is to ensure that all sequences have the
    same length.
    """
    for key, sequence in tracks_dict.iteritems():
      if len(sequence) < desired_seq_length:
        padding = [sequence[-1]] * (desired_seq_length - len(sequence))
        sequence.extend( padding )

  def crop_sequences(self, tracks_dict, desired_seq_length):
    for key, sequence in tracks_dict.iteritems():
      if len(sequence) > desired_seq_length:
        sequence = sequence[:desired_seq_length]

  def normalize_sequences(self, tracks_dict, desired_seq_length):
    dict_ = copy.deepcopy(tracks_dict)
    self.pad_sequences(dict_, desired_seq_length)
    self.crop_sequences(dict_, desired_seq_length)
    return dict_

  def check_sequences_have_equal_length(self, tracks_dict):
    shortest_len_seen = min([len(value) for value in tracks_dict.values()])


  def get_full_key(self, dict, player_id):
    full_key = None

    for key in dict.keys():
      if player_id in key:
        full_key = key

    if not full_key:
      LOG("player_id", player_id, "not found!", dict.keys())

    return full_key

  def get_macro_goal(self, track_xyzt, timestep, num_steps_ahead):
    """
    timestep: current timestep
    horizon: how far ahead we use to project
    """
    macro_goal = -1

    if len(track_xyzt) >= 2:
      if timestep < len(track_xyzt) - 1 and timestep >= 0:
        x = track_xyzt[timestep][0]
        y = track_xyzt[timestep][1]
        vx = track_xyzt[timestep + 1][0] - x
        vy = track_xyzt[timestep + 1][1] - y
        x_goal = x + num_steps_ahead * vx
        y_goal = y + num_steps_ahead * vy

        # curr = self.xy_to_serial_col_maj(x=x, y=y, x_range=self.phys_range_x, y_range=self.phys_range_y, x_scale=MG_GRID_X_CELLSIZE, y_scale=MG_GRID_Y_CELLSIZE)
        macro_goal = self.xy_to_serial_col_maj(x=x_goal, y=y_goal, x_range=MG_GRID_X_MAX, y_range=MG_GRID_Y_MAX, x_scale=MG_GRID_X_CELLSIZE, y_scale=MG_GRID_Y_CELLSIZE, verbose=False)
        # LOG("xy:", x, y, vx, vy, x_goal, y_goal, "mg:", curr, "->", macro_goal, "/", MG_GRID_X_MAX / MG_GRID_X_CELLSIZE * MG_GRID_Y_MAX / MG_GRID_Y_CELLSIZE)

    return macro_goal

  def get_possession_macrogoals(self, track_history_xyzt, player_id):
    """Given a possession, generate for every frame a macro-goal, a position that the
    player will reach if he does not change his velocity.

    Returns:
    - a list of len seq_len, each element is a serial_indexself.
    """
    track_xyzt = None
    for key, track in track_history_xyzt.iteritems():
      if key[2] == player_id: # key is (possession_id, team_id, player_id)
        track_xyzt = track
        break

    if track_xyzt and len(track_xyzt) > 0:

      macro_goal_labels = []
      horizon           = FLAGS.num_frames_ahead_for_macrogoal

      for timestep, idx in enumerate(track_xyzt[:-1]):
        macro_goals_this_frame = []

        macro_goal_label = self.get_macro_goal(track_xyzt, timestep, horizon / 2)
        macro_goals_this_frame.append( macro_goal_label )

        macro_goal_label = self.get_macro_goal(track_xyzt, timestep, horizon)
        macro_goals_this_frame.append( macro_goal_label )

        macro_goal_labels.append(macro_goals_this_frame)

      # For the last timestep, we copy the macro-goal from the penultimate timestep.
      # Be careful:
      if len(macro_goal_labels) > 1:
        macro_goal_labels.append( macro_goal_labels[-1] )


      return macro_goal_labels
    else:

      LOG("Player", player_id, " not found in track history! Keys:", track_history_xyzt.keys(), "No macro-goals constructed, but plugged with -1s.")
      return [-1] * len(track_history_xyzt.values()[0])

  def get_track_segments(self, track_history_xyzt, player_id):
    key = self.get_full_key(track_history_xyzt, player_id)

    if not key:
      return None

    segments = []

    V_THRESHOLD     = 0.25 # * constants.domain_scale
    # YY's threshold was for 1ft x 1ft cells. Since we're using floats here for the segmentation, cell scale doesn't matter.

    MIN_SEGMENT_LEN = 15

    segment_id = 0
    segment_len = 0

    track_len = len(track_history_xyzt[key])

    x = numpy.array( [track_history_xyzt[key][t] for t in xrange(track_len) ] )
    v = numpy.zeros((track_len, 4), dtype=numpy.float32)

    for t in xrange(track_len - 1):
      v[t] = x[t+1] - x[t]
      norm_v = math.sqrt(v[t][:3].dot(v[t][:3])) # only compute norm of spatial components!

      # LOG("segm: t", t, "v", v[t], norm_v)

      if norm_v < V_THRESHOLD:
        segments.append(1)
      else:
        segments.append(0)

    if len(segments) == 0:
      LOG("empty segment:", segments, key, x, v)

    if not segments:
      segments = [0]
    else:
      segments.append(segments[-1])

    # LOG("segm:", segments)

    return segments

  def get_macrogoals_from_segments(self, track_xyt_sparse, key):
    """Given a possession, generate for every frame a macro-goal, a position that the
    player will reach if he does not change his velocity.

    Returns:
    - a list of len seq_len, each element is a serial_indexself.
    """
    if key in track_xyt_sparse.keys():

      track = track_xyt_sparse[key]

      segment = self.segments[key]

      track_len = len(track)

      macro_goal_labels = []

      # Go through track in reverse, since macrogoal are the last position in a segment.
      macro_goal = track[-1]
      # LOG("Start with macro-goal", macro_goal)

      # LOG("track:", track)

      for timestep in xrange(track_len):

        macro_goals_this_frame = []

        # if this is a breakpoint ([ ..., 1, 0, ...]) then switch macro-goals
        if segment[-1-timestep] != 0 and segment[-timestep] == 0:
          macro_goal = track[-1-timestep]
          # LOG("Change to macro-goal", track_sparse_idx[timestep-1:timestep+2])
        # LOG(timestep, "macro-goal old:", macro_goal)

        # macro_goals_this_frame.append( macro_goal )


        # Get x,y coordinates in tracking grid
        x,y = self.serial_to_xy_col_maj(macro_goal, rows=constants.domain_height, cols=constants.domain_width)
        # LOG("Macro-goal in tracking grid", macro_goal, "xy:", x, y)

        # Get x,y, coordinates in the macro-goal grid.
        # WARNING: we scale twice! 1. from tracking grid to 50x45 2) from 50x45 to macro-goal grid.
        x,y = int(x / float(self.scale_domain_by * constants.macro_domain_scale)), int(y / float(self.scale_domain_by * constants.macro_domain_scale))

        # LOG("Macro-goal in mg grid", macro_goal, "xy:", x, y, "mg grid", constants.mg_grid_x_max, constants.mg_grid_y_max)
        # Compute new serial_index in macro-goal grid (macro-goal boxes)
        macro_goal_new = self.xy_to_serial_col_maj(x=x,
                                                   y=y,
                                                   x_range=constants.mg_grid_x_max,
                                                   y_range=constants.mg_grid_y_max, verbose=False)

        macro_goals_this_frame.append( macro_goal_new )

        # LOG("Macro-goal", x,y, "-->", macro_goal_new)

        macro_goal_labels.append(macro_goals_this_frame)

      # we went in reverse, so we need to flip the array
      macro_goal_labels.reverse()

      # LOG("track_history_xyzt", self.track_history_xyzt[key])
      # LOG("track_sparse_idx", track_sparse_idx)
      # LOG("segment", segment)
      # LOG("macro_goal_labels", macro_goal_labels)

      return macro_goal_labels

    else:

      LOG("Key", key, " not found in track history! Keys:", track_xyt_sparse.keys(), "No macro-goals constructed, but plugged with -1s.")
      return [-1] * len(track_xyt_sparse.values()[0])

  def adjust_macro_goals_for_subsequence_tail(self):
    """We typically take subsequences of the full possession sequence, so we set the macro-goal of the last segment to be the
    last position in the subsequence. This avoids spurious macro-goals that the subsequence never reaches, but the full sequences does."""

    x,y = self.serial_to_xy_col_maj(macro_goal, rows=constants.domain_height, cols=constants.domain_width)
    # LOG("Macro-goal 200x180 grid", macro_goal, "xy:", x, y)

    # Get x,y, coordinates in standard 50 x 45 grid.
    x,y = x / 4, y / 4

    # Compute new serial_index in 10 x 9 grid (macro-goal boxes)
    macro_goal_new = self.xy_to_serial_col_maj(x=x,
                                               y=y,
                                               x_range=constants.mg_grid_x_max,
                                               y_range=constants.mg_grid_y_max, verbose=False)
    macro_goals_this_frame.append( macro_goal_new )


  # Write sparse indices to binary file
  def write_tracks(self, offense_team, defense_team, possession_id, ballhandler_id):

    len_shortest_seq_seen = min([len(i) for i in self.tracks_vel_xyt.values()])

    raw_length = len(self.ball_xyzt.values()[0])

    if FLAGS.encoder_input_length > 0:
      self.pad_sequences(self.tracks_xyt, FLAGS.encoder_input_length)
      self.pad_sequences(self.ball_xyzt, FLAGS.encoder_input_length)
      self.pad_sequences(self.tracks_vel_xyt, FLAGS.encoder_input_length)
      self.pad_sequences(self.labels, FLAGS.encoder_input_length)
    elif len_shortest_seq_seen > 0:
      self.pad_sequences(self.tracks_xyt, len_shortest_seq_seen)
      self.pad_sequences(self.ball_xyzt, len_shortest_seq_seen)
      self.pad_sequences(self.tracks_vel_xyt, len_shortest_seq_seen)
      self.pad_sequences(self.labels, len_shortest_seq_seen)
    else:
      LOG("Not enough frames seen, len_shortest_seq_seen:", len_shortest_seq_seen, "for seq of len", FLAGS.encoder_input_length)
      return

    # Only write tracks if we have enough labels
    num_timesteps = len_shortest_seq_seen

    sample_key = (possession_id, offense_team, ballhandler_id)
    team_1 = offense_team
    team_2 = defense_team

    if sample_key not in self.tracks_vel_xyt.keys():

      sample_key = (possession_id, defense_team, ballhandler_id)

      if sample_key not in self.tracks_vel_xyt.keys():
        LOG("Sample not correct:", sample_key, self.tracks_vel_xyt.keys(), "data must be corrupted / not correct. (bh id not corrent in meta-data.")
        return
      else:
        team_2 = offense_team
        team_1 = defense_team
        # LOG("team_id_1", team_id_1)
        # LOG("team_id_2", team_id_2)
        # LOG("teamA_direction", teamA_direction)
        # LOG("possession", possession)
        # LOG("offense_team", offense_team)
        # LOG("defense_team", defense_team)

    if sample_key in self.tracks_xyt.keys() and \
       sample_key in self.tracks_vel_xyt.keys() and \
       sample_key in self.labels.keys():

      # Write tracks
      if all([len(i) >= FLAGS.encoder_input_length for i in self.tracks_vel_xyt.values()]):

        if FLAGS.use_multiple_start_times:
          starting_times = xrange(0, num_timesteps - FLAGS.encoder_input_length - 1 - FLAGS.num_frames_ahead_for_microlabels, 25)
        else:
          starting_times = [0]

        for starting_timestep in starting_times:

          # LOG(sample_key, "writing track with", FLAGS.encoder_input_length, "steps, starting at", starting_timestep, "/", len_shortest_seq_seen)

          self.write_sparse_tracks_and_label_to_bin(possession_id, team_1, team_2, ballhandler_id,
                                                    self.tracks_xyt, self.ball_xyzt, self.tracks_vel_xyt, self.labels,
                                                    starting_timestep,
                                                    raw_length, FLAGS.subsample_steps)
          # LOG("Fliping offense / defense, as a hack, b/c direction / team ids is not consistent yet.")
          # LOG("ballhandler_ids seen:", ballhandler_ids)
          # LOG("Skipping this possesion... key not found in velocity:", sample_key, tracks_vel_xyt.keys(), "did we mirror?", mirror)
    else:
      LOG("Key not correct -- something went wrong", sample_key)
      LOG("self.tracks_xyt.keys()", self.tracks_xyt.keys())
      LOG("self.tracks_vel_xyt.keys()", self.tracks_vel_xyt.keys())
      LOG("self.ball_xyzt.keys()", self.ball_xyzt.keys())
      LOG("self.labels.keys()", self.labels.keys())


  # TF readers read a set of bytes for each data sample. Therefore, concatenate input and label.
  def write_sparse_tracks_and_label_to_bin(self, possession, offense_team, defense_team, ballhandler_id,
    tracks_xyt, ball_xyzt, tracks_vel_xyt, labels, starting_timestep, raw_length, subsample_steps):

    if FLAGS.encoder_input_length > 0:
      input_sequence_length = FLAGS.encoder_input_length
      dir_prefix = "possessions_maxlen%i" % input_sequence_length
    else:
      input_sequence_length = min([len(i) for i in tracks_vel_xyt.values()]) # note that velocity / actions have 1 frame less (we computed diffs). That's why it determines the sequence length.
      dir_prefix = "possessions_varlen"

    output_dir = os.path.join(FLAGS.output_dir, dir_prefix, "team-%i" % (offense_team))
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    with open(os.path.join(output_dir, "team-%i-poss-%i-maxlen-%i-rawlen-%i-sub-%i-start-%i-macro-%i-micro-%i.bin" % (offense_team, possession, input_sequence_length, raw_length, subsample_steps, starting_timestep, FLAGS.num_frames_ahead_for_macrogoal, FLAGS.num_frames_ahead_for_microlabels)), "wb") as src_outfile:

      assert starting_timestep + input_sequence_length <= len(ball_xyzt[(possession)]), "not enough data-points!"

      possession_len = len(ball_xyzt[(possession)])

      print "Writing to", src_outfile

      # Write up to input_sequence_length timesteps, or raw_length, whichever comes first.
      for timestep in xrange(starting_timestep, starting_timestep + min(raw_length, input_sequence_length), subsample_steps):

        # Ball
        key = (possession)
        bytes = struct.pack("i", int(ball_xyzt[key][timestep]))
        src_outfile.write(bytes)

        # Ballhandler
        key = (possession, offense_team, ballhandler_id)

        if timestep >= len(tracks_xyt[key]):
          LOG(timestep, len(tracks_xyt[key]), key, tracks_xyt[key])

        bytes = struct.pack("i", int(tracks_xyt[key][timestep]))
        src_outfile.write(bytes)

        # LOG("ball:", ball_xyzt[(possession)][timestep], ball_xyzt[(possession)][timestep] / 180, ball_xyzt[(possession)][timestep] % 180, "bh:", tracks_xyt[key][timestep], tracks_xyt[key][timestep] / 180, tracks_xyt[key][timestep] % 180)

        # Teammates
        for key, val in tracks_xyt.iteritems():
          if key[1] == offense_team and key[2] != ballhandler_id:

            if timestep >= len(tracks_xyt[key]):
              LOG(timestep, len(tracks_xyt[key]), key, tracks_xyt[key])

            bytes = struct.pack("i", int(tracks_xyt[key][timestep]))
            src_outfile.write(bytes)

        # Defenders
        for key, val in tracks_xyt.iteritems():
          if key[1] == defense_team:

            if timestep >= len(tracks_xyt[key]):
              LOG(timestep, len(tracks_xyt[key]), key, tracks_xyt[key])

            bytes = struct.pack("i", int(tracks_xyt[key][timestep]))
            src_outfile.write(bytes)

        # Micro-goals
        # We want to predict the next actions (velocities) of the ballhandler
        key = (possession, offense_team, ballhandler_id)

        num_micro_labels = FLAGS.num_frames_ahead_for_microlabels

        # If we reach the end of the sequence, we replicate the velocity labels for the future frames that are not in the data.
        micro_labels = [tracks_vel_xyt[key][min(timestep + i, possession_len - 1)] for i in range(0, num_micro_labels)]
        bytes = struct.pack("%si" % num_micro_labels, *micro_labels)
        src_outfile.write(bytes)

        # The second label is the macro-goal of the player
        if FLAGS.num_macro_labels > 0:
          key = (possession, offense_team, ballhandler_id)


          assert FLAGS.num_macro_labels == len(labels[key][timestep])
          macro_labels = labels
          bytes = struct.pack("%si" % FLAGS.num_macro_labels, *macro_labels[key][timestep])
          src_outfile.write(bytes)

          # print "macro", [len(i) for i in labels.values()], timestep, input_sequence_length

if FLAGS.parse_simple:
  data_dir = FLAGS.simple_data_dir
else:
  data_dir = FLAGS.full_data_dir


parser = Parser(data_dir=data_dir, output_dir=FLAGS.output_dir, parse_simple=FLAGS.parse_simple)

def func(team_dirs, p=Parser()):
  return p.parse_full_data_as_fixlen_raw(team_dirs)

def run_functor(functor):
  """
  Given a no-argument functor, run it and return its result. We can
  use this with multiprocessing.map and map it over a list of job
  functors to do them.

  Handles getting more than multiprocessing's pitiful exception output
  """

  try:
    # This is where you do your actual work
    return functor()
  except:
    # Put all exception text into an exception and raise that
    raise Exception("".join(traceback.format_exception(*sys.exc_info())))


os.chdir(data_dir)
team_dirs = os.listdir(os.getcwd())

for i in xrange(len(team_dirs)):
  id_queue.put(i)

print "team_dirs", team_dirs
pool = Pool(processes=FLAGS.num_threads)


pool.map_async(func, team_dirs).get(9999999)


try:
  LOG("All processes done!")
except KeyboardInterrupt:
  pool.terminate()
  pool.join()
else:
  print "Quitting normally"
  pool.close()
  pool.join()


# parser.do_parsing()

