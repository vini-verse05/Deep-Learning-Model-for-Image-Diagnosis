from utils.preprocessing import get_data_generators

train_gen, test_gen = get_data_generators()

print(train_gen.samples)
print(test_gen.samples)