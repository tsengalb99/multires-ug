def get_params(args):

  print("Getting params for", args.algo, "for model", args.model_type)

  if args.algo == "multires":
    _params = { "type": args.model_type,
                "epochs": args.epochs,}
  return _params
