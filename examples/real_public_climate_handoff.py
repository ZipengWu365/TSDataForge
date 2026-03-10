from tsdataforge import demo

bundle = demo(output_dir="climate_bundle", scenario="climate_public")
print(bundle.output_dir)
print(bundle.index.recommended_next_step)
