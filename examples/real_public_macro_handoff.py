from tsdataforge import demo

bundle = demo(output_dir="macro_bundle", scenario="macro_public")
print(bundle.output_dir)
print(bundle.index.recommended_next_step)
