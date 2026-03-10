from tsdataforge import demo

bundle = demo(output_dir="ecg_bundle", scenario="ecg_public")
print(bundle.output_dir)
print(bundle.index.to_min_dict())
