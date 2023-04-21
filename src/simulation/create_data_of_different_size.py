from src.data_generation.data_generator import DataGenerationStrategy, DataGenerator, additive_target_fct, logit_random_link_fct
from src.data_generation.term import Term, TermFactory
from src.data_generation.distribution import NormalDist
from src.util.data_in_out import write_data, reset_data_storage
from src.util.timer import Timer


def create_data(n):
    coeff_dist = NormalDist({"loc": 0, "scale": 1})
    var_dist = NormalDist({"loc": 0, "scale": 1})

    tf = TermFactory('X')
    terms = tf.produce_n(input=(coeff_dist, var_dist), n=5)

    dgs = DataGenerationStrategy(terms=terms, 
                                target_fct=additive_target_fct, 
                                link_fct= logit_random_link_fct)
    
    generator = DataGenerator()
    features, target, transformed_target = generator.generate_data(dgs, n = n)
    features['transformed_target'] = transformed_target
    return features


def main():
    t = Timer()
    t.start()
    samples_sizes = [10**i for i in range(1, 6)]
    for n in samples_sizes: 
        data = create_data(n)
        write_data(data, f"size_{n}.csv")
    t.end()
    print(t.duration)
    
reset_data_storage()
main()

