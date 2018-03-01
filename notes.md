# Notes

### Struct-RL

### Multires

Text

- cut cut cut
- table:
  - bb fixedres parafac -- running on cms
- fvf graph
- fvf code fix?
- figure caption




  Handhold runs:
  3. bb full mu-sd @ cms

  3. bb fact @ cms
    - RUN fixedres?
    - Get more ent / sd_div runs?


  3. fvf fact - aws
    - loss_conv, fixed ok
    - need good ent_div, mu_sd, sd_div runs
    if needed, manually RERUN on CMS for more exps


  fvf has a bug ....
  2. check fvf full - cms
    - running now 8.30am

  4. Collate experiments.
    - Make new loss curve vs time (replace old one)
      - bb full, fact
      - fvf full, fact
    - sensitivity
      - bb full






figs = []
keep_fp = []
keep_best = []

# fvf, full

keep_fp += [
    "/tmp/stephan/logs/multires/group2/multires_fvf_200000_tt0.90/full/loss_conv/10-13-17_03-56-37/run_3",
    "/tmp/stephan/logs/multires/group2/multires_fvf_200000_tt0.90/full/loss_conv/10-13-17_03-56-37/run_2",



    "/tmp/stephan/logs/multires/group2/multires_fvf_200000_tt0.90/full/sd_div/10-13-17_03-56-43/run_1",
]


# fvf, parafac

keep_fp += [
  "/tmp/stephan/logs/multires/group2/multires_fvf_200000_tt0.90/parafac/fixedres/10-13-17_02-48-08/run_1",

  "/tmp/stephan/logs/multires/group2/multires_fvf_200000_tt0.90/parafac/loss_conv/10-12-17_17-09-56/run_2",
  "/tmp/stephan/logs/multires/group2/multires_fvf_200000_tt0.90/parafac/loss_conv/10-12-17_17-09-56/run_3",
  "/tmp/stephan/logs/multires/group3/multires_fvf_200000_tt0.90/parafac/loss_conv/10-13-17_11-01-32/run_1",


  "/tmp/stephan/logs/multires/group3/multires_fvf_200000_tt0.90/parafac/ent_div/10-13-17_11-01-53/run_1",

]

# bb, full
keep_fp += [
  "/tmp/stephan/logs/multires/group2/multires_bb_200000_sw0.50_tt0.90/full/ent_div/10-12-17_22-03-15/run_2",
  "/tmp/stephan/logs/multires/group2/multires_bb_200000_sw0.50_tt0.90/full/ent_div/10-12-17_22-03-15/run_1",
  "/tmp/stephan/logs/multires/group2/multires_bb_200000_sw0.50_tt0.90/full/ent_div/10-13-17_03-56-26/run_1",
  "/tmp/stephan/logs/multires/group2/multires_bb_200000_sw0.50_tt0.90/full/ent_div/10-13-17_03-56-30/run_1",

  "/tmp/stephan/logs/multires/group2/multires_bb_200000_sw0.50_tt0.90/full/sd_div/10-13-17_03-56-15/run_1",
  "/tmp/stephan/logs/multires/group2/multires_bb_200000_sw0.50_tt0.90/full/sd_div/10-12-17_22-03-17/run_2",

  "/tmp/stephan/logs/multires/group2/multires_bb_200000_sw0.50_tt0.90/full/mu_sd/10-12-17_18-22-42/run_1",
  "/tmp/stephan/logs/multires/group2/multires_bb_200000_sw0.50_tt0.90/full/mu_sd/10-13-17_03-56-18/run_1",

  "/tmp/stephan/logs/multires/group2/multires_bb_200000_sw0.50_tt0.90/full/loss_conv/10-12-17_15-03-46/run_1",
  "/tmp/stephan/logs/multires/group2/multires_bb_200000_sw0.50_tt0.90/full/loss_conv/10-13-17_03-56-06/run_1",
  "/tmp/stephan/logs/multires/group2/multires_bb_200000_sw0.50_tt0.90/full/loss_conv/10-12-17_22-02-47/run_1",

  "/tmp/stephan/logs/multires/group2/multires_bb_200000_sw0.50_tt0.90/full/fixedres/10-13-17_03-56-03/run_1",

]

# bb, parafac
keep_fp += [
  "/tmp/stephan/logs/multires/group2/multires_bb_200000_sw0.50_tt0.90/parafac/loss_conv/10-12-17_21-26-34/run_3",
  "/tmp/stephan/logs/multires/group2/multires_bb_200000_sw0.50_tt0.90/parafac/loss_conv/10-12-17_21-26-34/run_1",
  "/tmp/stephan/logs/multires/group2/multires_bb_200000_sw0.50_tt0.90/parafac/loss_conv/10-12-17_21-20-20/run_1",
  "/tmp/stephan/logs/multires/group3/multires_bb_200000_sw0.50_tt0.90/parafac/loss_conv/10-12-17_21-26-34/run_1",
  "/tmp/stephan/logs/multires/group3/multires_bb_200000_sw0.50_tt0.90/parafac/loss_conv/10-12-17_21-26-34/run_2",
  "/tmp/stephan/logs/multires/group3/multires_bb_200000_sw0.50_tt0.90/parafac/loss_conv/10-12-17_21-26-34/run_3",


  "/tmp/stephan/logs/multires/group2/multires_bb_200000_sw0.50_tt0.90/parafac/ent_div/10-12-17_21-36-47/run_3",
  "/tmp/stephan/logs/multires/group3/multires_bb_200000_sw0.50_tt0.90/parafac/ent_div/10-12-17_21-36-47/run_3",

  "/tmp/stephan/logs/multires/group3/multires_bb_200000_sw0.50_tt0.90/parafac/mu_sd/10-13-17_11-01-22/run_8",
  "/tmp/stephan/logs/multires/group3/multires_bb_200000_sw0.50_tt0.90/parafac/mu_sd/10-13-17_11-01-19/run_1",

  "/tmp/stephan/logs/multires/group3/multires_bb_200000_sw0.50_tt0.90/parafac/sd_div/10-12-17_21-27-06/run_5",
  "/tmp/stephan/logs/multires/group2/multires_bb_200000_sw0.50_tt0.90/parafac/sd_div/10-12-17_21-27-06/run_5",



  "/tmp/stephan/logs/multires/group3/multires_bb_200000_sw0.50_tt0.90/parafac/ent_div/10-12-17_21-36-47/run_3",
  "/tmp/stephan/logs/multires/group3/multires_bb_200000_sw0.50_tt0.90/parafac/ent_div/10-12-17_21-36-47/run_1",
  "/tmp/stephan/logs/multires/group3/multires_bb_200000_sw0.50_tt0.90/parafac/ent_div/10-12-17_21-36-47/run_2",
  "/tmp/stephan/logs/multires/group2/multires_bb_200000_sw0.50_tt0.90/parafac/ent_div/10-12-17_21-36-47/run_3",
  "/tmp/stephan/logs/multires/group2/multires_bb_200000_sw0.50_tt0.90/parafac/ent_div/10-12-17_21-36-47/run_1",
  "/tmp/stephan/logs/multires/group2/multires_bb_200000_sw0.50_tt0.90/parafac/ent_div/10-13-17_02-48-05/run_3",
  "/tmp/stephan/logs/multires/group2/multires_bb_200000_sw0.50_tt0.90/parafac/ent_div/10-13-17_01-11-52/run_3",


  "/tmp/stephan/logs/multires/group2/multires_bb_200000_sw0.50_tt0.90/parafac/sd_div/10-12-17_21-27-06/run_5",
  "/tmp/stephan/logs/multires/group2/multires_bb_200000_sw0.50_tt0.90/parafac/sd_div/10-12-17_21-27-06/run_3",
  "/tmp/stephan/logs/multires/group2/multires_bb_200000_sw0.50_tt0.90/parafac/sd_div/10-13-17_00-29-52/run_1",
  "/tmp/stephan/logs/multires/group3/multires_bb_200000_sw0.50_tt0.90/parafac/sd_div/10-12-17_21-27-06/run_5",


  "/tmp/stephan/logs/multires/group3/multires_bb_200000_sw0.50_tt0.90/parafac/mu_sd/10-13-17_11-01-19/run_1",
  "/tmp/stephan/logs/multires/group3/multires_bb_200000_sw0.50_tt0.90/parafac/mu_sd/10-13-17_11-01-19/run_4",
  "/tmp/stephan/logs/multires/group3/multires_bb_200000_sw0.50_tt0.90/parafac/mu_sd/10-13-17_11-01-15/run_5",
  "/tmp/stephan/logs/multires/group3/multires_bb_200000_sw0.50_tt0.90/parafac/mu_sd/10-13-17_11-01-22/run_8",
]





keep_best += [
  "/tmp/stephan/logs/multires/group2/multires_bb_200000_sw0.50_tt0.90/parafac/loss_conv/10-12-17_21-26-34/run_3",

  "/tmp/stephan/logs/multires/group2/multires_bb_200000_sw0.50_tt0.90/parafac/ent_div/10-12-17_21-36-47/run_3",

  "/tmp/stephan/logs/multires/group2/multires_bb_200000_sw0.50_tt0.90/parafac/sd_div/10-12-17_21-27-06/run_5",

]

















"/tmp/stephan/logs/multires/group2/multires_bb_200000_sw0.50_tt0.90/parafac/loss_conv/10-12-17_21-26-34/run_3",
"/tmp/stephan/logs/multires/group2/multires_bb_200000_sw0.50_tt0.90/parafac/loss_conv/10-12-17_21-26-34/run_1",
"/tmp/stephan/logs/multires/group2/multires_bb_200000_sw0.50_tt0.90/parafac/loss_conv/10-12-17_21-20-20/run_1",


Experiments

- Plot loss over time
- Sensitivity analysis: performance of various stopping criteria
  - Plot x=tau y= average loss curve (collate from several runs)
  - for bb, full, (fact)?
- Table


Tweaks tried

- Exps do NOT copy OPT state.
- bb with 5% quantile... more stable behavior?
- Try without lr decay
- Normalize loss by lr when comparing -- threshold more stable


  0. Plotting code -- ok
  0. Check 10pm CMS / AWS
    - bb full -- cms
    - bb full -- aws
      - ok, but stage 3 is very unstable... --> has converged already...?
      - loss_conv: need more runs

Bugs
- !!! lr=0 leftover from debugging... restarting everything.
- !!! adjusted scalar reporting rate
- !!! fixed bug in upscaling. Now always load pth.tar, don't use model etc.

Good runs

ml3 - 131.215.140.70
- 10-08-17_20-33-42 Adam-multires FULL 1+1+1 epochs 76% fixed transition time
- 10-06-17_11-49-58 Adam-multires FACT 1+1+1+7 epochs 74% fixed transition time
- 10-06-17_15-02-26 Adam-multires FACT 1+1+1+7 epochs 74% fixed transition time




### AA+YY

todo

- fix sampling of data
- add [x h] as aug state in LSTM (should be able to get )
- use kernel-TT (has guarantees)
- use dataset from explicitly finite-difference higher-order Markov process, show it's better than LSTM
- Lorenz is (too) hard: data skewedness, resolution.
- Genz function: investigate the errors
- Full-model Tensor (no fact): perf about the same.
- how does performance degrade from T = 10 20 30 40 50 ... when does the difference grow between models?
- moving window: subsample, make sure samples use offset > 1 between them, do not uniformly sample from every state.
