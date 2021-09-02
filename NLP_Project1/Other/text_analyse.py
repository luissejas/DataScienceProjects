import textstat as ts
import pandas as pd

def text_analyse(texts_list):
    fre_list = []
    smog_list = []
    fkg_list = []
    cli_list = []
    ari_list = []
    dcrs_list = []
    dw_list = []
    lwf_list = []
    gf_list = []
    t_std_list = []

    for text in texts_list:
        # Perform different tests from the textstat.ts package to score the
        # text's readability
        text = text.strip("\n")
        fre = ts.flesch_reading_ease(text)
        smog = ts.smog_index(text)
        fkg = ts.flesch_kincaid_grade(text)
        cli = ts.coleman_liau_index(text)
        ari = ts.automated_readability_index(text)
        dcrs = ts.dale_chall_readability_score(text)
        dw = ts.difficult_words(text)
        lwf = ts.linsear_write_formula(text)
        gf = ts.gunning_fog(text)
        t_std = ts.text_standard(text, float_output=True)

        fre_list.append(fre)
        smog_list.append(smog)
        fkg_list.append(fkg)
        cli_list.append(cli)
        ari_list.append(ari)
        dcrs_list.append(dcrs)
        dw_list.append(dw)
        lwf_list.append(lwf)
        gf_list.append(gf)
        t_std_list.append(t_std)
    
    df_dict = {'fre': fre_list, 'smog': smog_list, 'fkg': fkg_list, 
               'cli': cli_list, 'ari': ari_list, 'dcrs': dcrs_list, 
               'dw': dw_list, 'lwf': lwf_list, 'gf': gf_list, 
               't_std': t_std_list}
    # Construct a dataframe to store the test results for the text list
    df = pd.DataFrame(data=df_dict)
        
    return df