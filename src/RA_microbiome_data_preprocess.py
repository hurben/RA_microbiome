#RA_microbiome_data_preprocess.py       #2021.04.21
#hur.benjamin@mayo.edu
#
#"Specific" input: RA Dataset for Kevin.csv
#create a datamatrix that is ready for machine learning
#
#2 class (1/0) = visit 2 (MCII_yes_no)
#448 features = visit 1 (CDAI~ )  

#output
#,sample1, sample2, sample3
#class,1,0,1
#feature,value,value,value

import pandas as pd


def binary_convert(string):
    return_value = 0
    if string == 'No' or string == 'no' or string == 'female':
        return_value = 0
    if string == 'Yes' or string == 'yes' or string == 'male':
        return_value = 1
    return return_value

if __name__ == "__main__":

	data_file = '../data/RA_microbiome_data_matrix.csv'
	data_df = pd.read_csv(data_file,index_col=0)
	r, c = data_df.shape

	feature_list = list(data_df.index.values)
	sample_list = list(data_df.columns.values)
	sub_sample_list = []
	binary_features = ['sex','csDMARDs', 'bDMARDs', 'Prednisone', 'Smoking_status']

	data_dict = {}
	#data_dict[sample, feature] = value
	#data_dict[sample, class] = value

	i = 0
	while i < c:
		mcii_class = data_df.iloc[1][i+1] #mcii from t2
		mcii_class = binary_convert(mcii_class)
		sample = sample_list[i]
		sub_sample_list.append(sample)
		data_dict[sample, 'class'] = mcii_class
		for j in range(2, r):
			value_t1 = data_df.iloc[j][i]
			feature = feature_list[j]
			if feature in binary_features:
				value_t1 = binary_convert(value_t1)
			data_dict[sample, feature] = value_t1
		i = i+2
		
	output_file = '../data/RA_microbiome_data.ml_ready.matrix.tsv'
	output_txt = open(output_file,'w')

	#header
	for sample in sub_sample_list:
		sample = sample.replace('RHB_RAM_', 'temp_')
		output_txt.write('\t%s' % sample)
	output_txt.write('\n')

	#main
	output_txt.write('class')
	for sample in sub_sample_list:
		output_txt.write('\t%s' % data_dict[sample,'class'])
	output_txt.write('\n')

	for feature in feature_list[2:]:
		output_txt.write(feature)
		for sample in sub_sample_list:
			output_txt.write('\t%s' % data_dict[sample,feature])
		output_txt.write('\n')
	output_txt.close()

else:
	None
