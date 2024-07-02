from runway_for_ml.data_module.data_transforms import register_transform_functor, BaseTransform
import sys
sys.path.append('third_party')
from google_nlg.ser import get_ser_slots
import sacrebleu
import datasets
from data_ops.metrics import compute_ser, DARTExactOccurRateEvaluator
from pprint import pprint
import pandas as pd
import os
import json
import string

import logging
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

@register_transform_functor
class GetGEMSGDTestReferences(BaseTransform):
    def setup(self):
        pass
    
    def _call(self, data):
        sgd_ds = datasets.load_dataset('GEM/schema_guided_dialog', split='test')
        return {'references': sgd_ds['target']}

@register_transform_functor
class GetGEMDARTTestReferences(BaseTransform):
    def setup(self):
        pass

    def _call(self, data):
        dart_ds = datasets.load_dataset('GEM/dart', split='test')
        return {'references': dart_ds['target']}

@register_transform_functor
class ComputeBLEU(BaseTransform):
    """_summary_
    
    example config:

    'process:ComputeBLEUScore': {
      input_node: ['input:GetGEMSGDTestReferences', 'input:GetInferEvalRecorder'],
      transform_name: 'ComputeBLEU',
      setup_kwargs: {},
      regenerate: false,
      cache: false,
    },

    :param BaseTransform: _description_
    """
    def setup(self):
        pass

    def _call(self, data, *args, **kwargs):
        """data must contain keys 'eval_recorder' and 'references', other keys are optional

        :param data: _description_
        :return: _description_
        """
        refs = data[0]['references']
        eval_recorder = data[1]
        hypos = eval_recorder.get_sample_logs_column('prediction')
        no_bos_hypos = [hypo.replace("<s> ", "") for hypo in hypos] # remove hypothesis
        inps = eval_recorder.get_sample_logs_column('inps')
        bos_refs = eval_recorder.get_sample_logs_column('reference')

        setence_bleu_score = self.compute_sentence_bleu(hypos, refs)
        with_bos_setence_bleu_score = self.compute_sentence_bleu(hypos, bos_refs)
        remove_bos_sentence_bleu_score = self.compute_sentence_bleu(no_bos_hypos, refs)
        corpus_bleu_score = self.compute_corpus_bleu(hypos, refs)
        example_corpus_bleu_score = self.compute_example_level_corpus_bleu(hypos, refs)
        input_sentence_bleu = self.compute_sentence_bleu(inps, refs)

        eval_recorder.log_stats_dict({'corpus_bleu': corpus_bleu_score})
        eval_recorder.log_stats_dict({'avg_sentence_bleu': sum(setence_bleu_score)/len(setence_bleu_score)})
        eval_recorder.log_stats_dict({'avg_example_corpus_bleu': sum(example_corpus_bleu_score)/len(example_corpus_bleu_score)})
        eval_recorder.log_stats_dict({'avg_input_setence_bleu': sum(input_sentence_bleu)/len(input_sentence_bleu)})
        eval_recorder.log_stats_dict({'avg_setence_bleu_with_bos': sum(with_bos_setence_bleu_score)/len(with_bos_setence_bleu_score)})
        eval_recorder.log_stats_dict({'avg_sentence_bleu_with_no_bos_hypo': sum(remove_bos_sentence_bleu_score)/len(remove_bos_sentence_bleu_score)})

        eval_recorder.set_sample_logs_column('bleu', setence_bleu_score)
        eval_recorder.set_sample_logs_column('bleu-references', refs)
        eval_recorder.set_sample_logs_column('corpus_bleu', example_corpus_bleu_score)
        eval_recorder.set_sample_logs_column('input_setence_bleu', input_sentence_bleu)
        eval_recorder.set_sample_logs_column('bleu-with_bos', with_bos_setence_bleu_score)
        eval_recorder.set_sample_logs_column('bleu-hypo_no_bos', remove_bos_sentence_bleu_score)
        return eval_recorder

    def compute_sentence_bleu(self, preds, refs):
        sentence_bleu_res = []
        for r, p in zip(refs, preds):
            sentence_bleu_res.append(sacrebleu.sentence_bleu(p, [r]).score)
        return sentence_bleu_res
    
    def compute_example_level_corpus_bleu(self, preds, refs):
        example_corpus_bleu = []
        for r, p in zip(refs, preds):
            example_corpus_bleu.append(sacrebleu.corpus_bleu([p], [[r]]).score)
        return example_corpus_bleu

    def compute_corpus_bleu(self, preds, refs):
        corpus_bleu = sacrebleu.corpus_bleu(preds, [[r] for r in refs])
        return corpus_bleu.score

@register_transform_functor
class ComputeBLEU_V2(BaseTransform):
    """_summary_
    
    example config:

    'process:ComputeBLEUScore': {
      input_node: ['input:GetGEMSGDTestReferences', 'input:GetInferEvalRecorder'],
      transform_name: 'ComputeBLEU',
      setup_kwargs: {},
      regenerate: false,
      cache: false,
    },

    :param BaseTransform: _description_
    """
    def setup(self):
        pass

    def _call(self, data, *args, **kwargs):
        """data must contain keys 'eval_recorder' and 'references', other keys are optional

        :param data: _description_
        :return: _description_
        """
        refs = data[0]['references']
        eval_recorder = data[1]
        hypos = eval_recorder.get_sample_logs_column('prediction')
        no_bos_hypos = [hypo.replace("<s>", "").strip() for hypo in hypos] # remove hypothesis
        inps = eval_recorder.get_sample_logs_column('inps')

        setence_bleu_score = self.compute_sentence_bleu(no_bos_hypos, refs)
        input_sentence_bleu = self.compute_sentence_bleu(inps, refs)

        corpus_bleu_res = self.compute_corpus_bleu(no_bos_hypos, refs)
        input_corpus_bleu_res = self.compute_corpus_bleu(inps, refs)

        for bleu_field in ['score', 'bp', 'sys_len', 'ref_len']:
            eval_recorder.log_stats_dict({f'input_corpus_bleu_{bleu_field}': getattr(input_corpus_bleu_res, bleu_field)})
            eval_recorder.log_stats_dict({f'pred_corpus_bleu_{bleu_field}': getattr(corpus_bleu_res, bleu_field)})
        eval_recorder.log_stats_dict({'input_corpus_bleu_without_bp': input_corpus_bleu_res.score / input_corpus_bleu_res.bp if input_corpus_bleu_res.bp > 0.01 else 0.0})
        eval_recorder.log_stats_dict({'pred_corpus_bleu_without_bp': corpus_bleu_res.score / corpus_bleu_res.bp if input_corpus_bleu_res.bp > 0.01 else 0.0})

        eval_recorder.log_stats_dict({'avg_sentence_bleu': sum(setence_bleu_score)/len(setence_bleu_score)})
        pprint(eval_recorder.get_stats_logs())

        eval_recorder.set_sample_logs_column('bleu', setence_bleu_score)
        eval_recorder.set_sample_logs_column('bleu-references', refs)
        eval_recorder.set_sample_logs_column('input_setence_bleu', input_sentence_bleu)
        return eval_recorder

    def compute_sentence_bleu(self, preds, refs):
        sentence_bleu_res = []
        for r, p in zip(refs, preds):
            sentence_bleu_res.append(sacrebleu.sentence_bleu(p, [r]).score)
        return sentence_bleu_res

    def compute_corpus_bleu(self, preds, refs):
        print("Compute corpus bleu:", len(preds), len(refs))
        corpus_bleu = sacrebleu.corpus_bleu(preds, [refs])
        # corpus_bleu = sacrebleu.corpus_bleu(preds, [[r] for r in refs])
        return corpus_bleu

@register_transform_functor
class ComputeSER(BaseTransform):
    """
    example config:
    
    'process:ComputeSlotErrorRate': {
          input_node: ['input:GetInferEvalRecorder'],
          transform_name: 'ComputeSER'.
          setup_kwargs: {},
          regenerate: false,
          cache: false,
        },

    :param BaseTransform: _description_
    """
    def setup(self):
        self.sgd_test_split = datasets.load_dataset('GEM/schema_guided_dialog', split='test')
    
    def _call(self, data):
        eval_recorder = data
        preds = eval_recorder.get_sample_logs_column('prediction')
        sentence_ser, if_ser_applicable, corpus_ser, applicable_ser = compute_ser(turns=self.sgd_test_split, preds=preds)
        eval_recorder.set_sample_logs_column('sentence_ser', sentence_ser)
        eval_recorder.set_sample_logs_column('if_ser_applicable', if_ser_applicable)
        eval_recorder.log_stats_dict({'corpus_ser': corpus_ser, 'applicable_ser': applicable_ser})
        return eval_recorder

@register_transform_functor
class ComputeAverages(BaseTransform):
    def setup(self, fields_to_average, avg_field_prefix='avg_'):
        self.fields_to_average = fields_to_average
        self.avg_field_prefix = avg_field_prefix
    
    def _call(self, eval_recorder):
        for field in self.fields_to_average:
            values = eval_recorder.get_sample_logs_column(field)
            avg_value = sum(values) / len(values)
            eval_recorder.log_stats_dict({f"{self.avg_field_prefix}{field}": avg_value})
        return eval_recorder

@register_transform_functor
class DecodingDetailsAnalysis(BaseTransform):
    def setup(self):
        pass
    
    def _call(self, eval_recorder):
        decode_detail_infos = eval_recorder.get_sample_logs_column('decode_detail_infos')
        vocab_fail_cnt = 0
        sv_fail_cnt = 0
        length_fail_cnt = 0
        wfsa_fail_cnt = 0

        vocab_fail_col = [False for i in range(len(decode_detail_infos))]
        sv_fail_col = [False for i in range(len(decode_detail_infos))]
        length_fail_col = [False for i in range(len(decode_detail_infos))]
        wfsa_decode_size_col = []

        for i, detail_info in enumerate(decode_detail_infos):
            if detail_info.get('wfsa_decoding_fail_flag', False):
                wfsa_fail_cnt += 1
            if detail_info.get('wfsa_size_after_fsa_vocab_constraint', 1) == 0:
                vocab_fail_cnt += 1
                vocab_fail_col[i] = True
            if detail_info.get('wfsa_size_after_sv_constraints', 1) == 0:
                sv_fail_cnt += 1
                sv_fail_col[i] = True
            if detail_info.get('len_decode_success_flag', True) == False:
                length_fail_cnt += 1
                length_fail_col[i] = True
            if detail_info.get('wfsa_size_before_decoding', None):
                wfsa_decode_size_col.append(detail_info['wfsa_size_before_decoding'])
                

        eval_recorder.set_sample_logs_column('vocab_constraint_fail', vocab_fail_col)
        eval_recorder.set_sample_logs_column('slot_value_constraint_fail', sv_fail_col)
        eval_recorder.set_sample_logs_column('length_constraint_fail', length_fail_col)

        eval_recorder.log_stats_dict({
            "vocab_constraint_fail_cnt": vocab_fail_cnt,
            "slot_value_constraint_fail_cnt": sv_fail_cnt,
            "length_constraint_fail_cnt": length_fail_cnt,
            "wfsa_fail_cnt": wfsa_fail_cnt,
        })

        if len(wfsa_decode_size_col) > 0:
            eval_recorder.log_stats_dict({
                "avg_wfsa_decode_size": sum(wfsa_decode_size_col) / len(wfsa_decode_size_col),
                "max_wfsa_decode_size": max(wfsa_decode_size_col),
                "min_wfsa_decode_size": min(wfsa_decode_size_col),
            })

        sample_logs = eval_recorder.get_sample_logs()
        if 'out_degree_cnt' in sample_logs:
            out_degree_cnts = sample_logs['out_degree_cnt']
            avg_out_degree = sum([sum([(i+1)*oo for i, oo in enumerate(out_degrees)]) / sum(out_degrees) for out_degrees in out_degree_cnts]) / len(out_degree_cnts)
            eval_recorder.log_stats_dict({
                'avg_out_degree': avg_out_degree,
            })
        
        if 'emit_degree_cnt' in sample_logs:
            emit_degree_cnts = sample_logs['emit_degree_cnt']
            avg_emit_degree = sum([sum([(i+1)*ee for i, ee in enumerate(emit_degrees)]) / sum(emit_degrees) for emit_degrees in emit_degree_cnts]) / len(emit_degree_cnts)
            eval_recorder.log_stats_dict({
                'avg_emit_degree': avg_emit_degree,
            })

        return eval_recorder
        

@register_transform_functor
class DisplayEvalResults(BaseTransform):
    def setup(self, rows_to_display=5, display_format='csv'):
        self.rows_to_display = rows_to_display
        self.display_format = display_format
    
    def print_boarder(self):
        print("="*150)
    
    def _call(self, eval_recorder, *args, **kwargs):
        if self.display_format == 'csv':
            df = eval_recorder.get_sample_logs(data_format='csv')
            print("Available columns in sample logs:")
            pprint(df.columns.tolist())
            with pd.option_context('display.max_rows', self.rows_to_display, 'display.max_columns', None):
                print(df.head(n=self.rows_to_display))
        self.print_boarder()
        pprint(f"Full evaluation data saved to {eval_recorder.save_dir}")
        self.print_boarder()
        print(f"Evaluation Report for {self.global_config['experiment_name']}".center(150))
        self.print_boarder()
        pprint(eval_recorder.get_stats_logs(data_format='dict'))
        return eval_recorder

@register_transform_functor
class ComputeBLEURTScore(BaseTransform):
    def setup(self, checkpoint, *args, **kwargs):
        from bleurt import score as blt_score
        self.checkpoint = checkpoint
        print("Setup BLEURT: checkpoint=", self.checkpoint)
        self.bleurt_scorer = blt_score.LengthBatchingBleurtScorer(self.checkpoint)
    
    def _call(self, data, *args, **kwargs):
        refs = data[0]['references']
        eval_recorder = data[1]
        hypos = eval_recorder.get_sample_logs_column('prediction')
        inps = eval_recorder.get_sample_logs_column('inps')
        no_bos_hypos = [hypo.replace("<s> ", "") for hypo in hypos] # remove hypothesis

        scores = self.bleurt_scorer.score(references=refs, candidates=hypos)
        no_bos_scores = self.bleurt_scorer.score(references=refs, candidates=no_bos_hypos)
        inp_scores = self.bleurt_scorer.score(references=refs, candidates=inps)

        eval_recorder.set_sample_logs_column('bleurt', scores)
        eval_recorder.set_sample_logs_column('no_bos_bleurt', no_bos_scores)
        eval_recorder.set_sample_logs_column('input_bleurt', inp_scores)
        eval_recorder.log_stats_dict({'avg_bleurt': sum(scores)/len(scores)})
        eval_recorder.log_stats_dict({'avg_bleurt_no_bos': sum(no_bos_scores)/len(no_bos_scores)})
        eval_recorder.log_stats_dict({'avg_input_bleurt': sum(inp_scores)/len(inp_scores)})
        return eval_recorder

@register_transform_functor
class AnalyzeAndSaveBadCases(BaseTransform):
    def setup(self, case_num=20):
        self.case_num = case_num
        self.test_dir = f"{self.global_config['test_dir']}/qualitative_cases/"
        os.makedirs(self.test_dir, exist_ok=True)
        logger.debug(f"saving cases to {self.test_dir}")
    
    def _call(self, data, *args, **kwargs):
        eval_recorder = data
        sample_logs_data = eval_recorder.get_sample_logs()
        df = pd.DataFrame(sample_logs_data)

        #Save all cases
        df.to_csv(f'{self.test_dir}/all_cases.csv')

        #BLEU good/bad cases
        if 'sentence_bleu' in df.columns:
            bleu_sorted = df.sort_values(by='sentence_bleu', ascending=True)
            bleu_sorted.iloc[:self.case_num].to_csv(f'{self.test_dir}/bleu-low-cases.csv')
            bleu_sorted.iloc[-1:-self.case_num-1:-1].to_csv(f'{self.test_dir}/bleu-high-cases.csv')
            print(bleu_sorted.head(self.case_num))

        #BLEURT good/bad cases
        if 'bleurt' in df.columns:
            bleurt_sorted = df.sort_values(by='bleurt', ascending=True)
            bleurt_sorted.iloc[:self.case_num].to_csv(f'{self.test_dir}/bleurt-low-cases.csv')
            bleurt_sorted.iloc[-1:-self.case_num-1:-1].to_csv(f'{self.test_dir}/bleurt-high-cases.csv')
            print(bleurt_sorted.head(self.case_num))

        #Likelihood good/bad cases
        if 'decoded_score' in df.columns:
            llk_sorted = df.sort_values(by='decoded_score', ascending=True)
            llk_sorted.iloc[:self.case_num].to_csv(f'{self.test_dir}/llk-low-cases.csv')
            llk_sorted.iloc[-1:-self.case_num-1:-1].to_csv(f'{self.test_dir}/llk-high-cases.csv')
            print(llk_sorted.head(self.case_num))

        #decoded_time long/short cases
        if 'time_to_decode' in df.columns:
            dtime_sorted = df.sort_values(by='time_to_decode', ascending=False)
            dtime_sorted.iloc[:self.case_num].to_csv(f'{self.test_dir}/decode_time-long.csv')
            dtime_sorted.iloc[-1:-self.case_num-1:-1].to_csv(f'{self.test_dir}/decode_time-short.csv')
            print(dtime_sorted.head(self.case_num))
        
        #exact occur error (DART)
        if 'has_exact_occur_error' in df.columns:
            eor_sorted = df[df['has_exact_occur_error'] == True]
            eor_sorted.to_csv(f'{self.test_dir}/exact_occur_error.csv')
            print(eor_sorted.head(self.case_num))

        return data

@register_transform_functor
class ComputeNeologismRate(BaseTransform):
    def setup(self, all_vocab_file=None, lower_case=False, no_numeric=True, strip_punct=True, save_to_file=True, *args, **kwargs):
        if all_vocab_file is not None and os.path.exists(all_vocab_file):
            with open(all_vocab_file, 'r') as f:
                self.all_vocab_list = json.load(f)

        self.lower_case = lower_case
        if lower_case:
            self.all_vocab_list = [vocab.lower() for vocab in self.all_vocab_list]
        
        self.no_numeric = no_numeric
        self.strip_punct = strip_punct
        self.all_vocab_set = set(self.all_vocab_list)

        self.save_to_file = save_to_file

        if self.save_to_file:
            self.test_dir = os.path.join(self.global_config.get('test_dir'), 'neologism')
            os.makedirs(self.test_dir, exist_ok=True)
    
    def get_space_delimited_words(self, sentence, lower_case, no_numeric, strip_punct):
        # if strip_punct:
            # sentence = sentence.translate(str.maketrans(string.punctuation, " "*len(string.punctuation)))
            # sentence = sentence.translate(str.maketrans("", "", string.punctuation))
        words = sentence.split(' ')
        # if strip_punct:
            # words = [word.strip(string.punctation) for word in words]
            # words = [word.translate(str.maketrans(string.punctuation, " "*len(string.punctuation))) for word in words]
        if strip_punct:
            words = [word.strip(string.punctuation) for word in words]
        if no_numeric:
            words = [word for word in words if (len(word) and not word[0].isnumeric())]
        if lower_case:
            words = [word.lower() for word in words]
        return words
    
    def _call(self, eval_recorder, *args, **kwargs):
        from spellchecker import SpellChecker
        checker = SpellChecker()
        
        refs = eval_recorder.get_sample_logs_column('reference')
        inps = eval_recorder.get_sample_logs_column('inps')
        hypos = eval_recorder.get_sample_logs_column('prediction')
        neologism = []
        all_neo_words = set() 
        neo_word_cnt = []
        has_neo_word_list = []
        for inp, ref, hypo in zip(inps, refs, hypos):
            hypo = hypo.replace('<s>', '').replace('</s>', '') # strip BOS, EOS tokens
            ref = ref.replace('<s>', '').replace('</s>', '') # strip BOS, EOS tokens
            words_in_hypo = self.get_space_delimited_words(hypo, self.lower_case, self.no_numeric, self.strip_punct)
            words_in_ref = self.get_space_delimited_words(ref, self.lower_case, self.no_numeric, self.strip_punct)
            words_in_inp = self.get_space_delimited_words(inp, self.lower_case, self.no_numeric, self.strip_punct)
            example_neo_words = []
            example_neo_word_cnt = 0
            has_neo_word = False
            for word in words_in_hypo:
                if not word in self.all_vocab_set \
                    and len(checker.unknown([word])) > 0 \
                    and word not in words_in_ref \
                    and word not in words_in_inp:
                    example_neo_word_cnt += 1
                    example_neo_words.append(word)
                    all_neo_words.add(word)
                    has_neo_word = True
            has_neo_word_list.append(has_neo_word)
            neologism.append("|".join(example_neo_words))
            neo_word_cnt.append(example_neo_word_cnt)

        eval_recorder.set_sample_logs_column('neo_words', neologism)
        eval_recorder.set_sample_logs_column('neo_words_count', neo_word_cnt)
        eval_recorder.set_sample_logs_column('has_neo_word', has_neo_word_list)
        eval_recorder.log_stats_dict({
            'total_neo_word_count': len(all_neo_words),
            'has_neo_word_example_rate': sum(has_neo_word_list)/len(has_neo_word_list)
        })

        if self.save_to_file:
            save_filepath = os.path.join(self.test_dir, 'all_neo_words.json')
            with open(save_filepath, 'w') as f:
                json.dump(list(all_neo_words), f)
            print("Dumped all neo words to", save_filepath)

        if self.save_to_file:
            data = eval_recorder.get_sample_logs()
            df = pd.DataFrame(data)
            df.sort_values(by='neo_words_count', ascending=False).to_csv(os.path.join(self.test_dir, 'neo_all_cases.csv'))

        return eval_recorder
                    
@register_transform_functor
class ComputeDARTExactOccurErrorRate(BaseTransform):
    def setup(self, exact_occur_file):
        self.evaluator = DARTExactOccurRateEvaluator(exact_occur_file)
    
    def _call(self, eval_recorder):
        triples = eval_recorder.get_sample_logs_column('triples')
        hypos = eval_recorder.get_sample_logs_column('prediction')
        has_exact_occur_error = []
        all_forced_phrases = []
        eor_cnt = 0
        for triple, hypo in zip(triples, hypos):
            error, forced_phrases = self.evaluator.compute_eor(triple, hypo)
            all_forced_phrases.append(forced_phrases)
            if error:
                eor_cnt += 1
                has_exact_occur_error.append(True)
            else:
                has_exact_occur_error.append(False)

        eval_recorder.set_sample_logs_column('has_exact_occur_error', has_exact_occur_error)
        eval_recorder.set_sample_logs_column('forced_phrases', all_forced_phrases)
        eval_recorder.log_stats_dict({'exact_occurence_error': eor_cnt / len(hypos)})
        
        return eval_recorder