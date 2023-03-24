from taxoenrich.models import HypernymPredictModel
import argparse
import codecs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--input_path', required=True)
    parser.add_argument('--output_path', required=True)
    args = parser.parse_args()


    model = HypernymPredictModel.from_pretrained(args.model_dir)
    with codecs.open(args.input_path, 'r', 'utf-8') as file:
        words = [w.strip().lower() for w in file.read().strip().split('\n')]

    predict = model.predict(words)
    predict.to_csv(args.output_path, index=False, sep='\t')
    