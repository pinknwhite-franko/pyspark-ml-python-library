# Notebook: Column Vectorizer with Custom Parser
# Description: vectorize a string column using a custom parser with pyspark

from pyspark.sql.functions import col, udf, flatten, array_distinct, explode, array_contains, transform
from pyspark.sql.types import StringType, ArrayType
import shlex
import re
from pyspark.sql.window import Window
from pyspark.sql.functions import *
 



class StringColumnVectorizer:
    """ 
    This python class is used to vectorize a string column using a custom parser. compatible with pyspark 3.5.2
    1. Parse each string in the column into an array of tokens
    2. Create bag of words with those tokens (each token with a unique key)
    3. Vectorized arrays of tokens into array of int numbers
    4. Return the pyspark dataframe with the vectorized column
    """
    def __init__(self,inputCol, outputCol, uuid):
        """ instantiate an object upon call
        Args:
            inputCol:   name of the input parameter column
            outputCol:  name of the output parameter column
            uuid:       name of id column
        Returns:
            n/a
        """
        # Calling construct
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.uuid = uuid
        self.pyspark_dataframe_with_parsed_col= None
        self.parsed_column_name = self.inputCol  + '_parsed'
        self.dictionary= None
   
 
    # where you define your parser
    @staticmethod
    @udf(returnType=ArrayType(StringType()))
    def parser(inputString):
        """ static class that define the CUSTOM PARSER. This UDF has be staitc and will be pickled
        Args: 
            inputString:   the input string that will be parsed
        Returns:
            parsed_tokens: the parsed tokens from the input strings
        """

        """
        INPUT YOUR CODE HERE. or use the example code below
        """
        parsed_tokens = []
        for token in inputString.lower().split("\""):               # parse wth "
            for layer2_token in token.split("\'"):                  # parse wth '
                for layer3_token in layer2_token.split("\\"):       # parse wth \
                    for layer4_token in shlex.split(layer3_token):  # parse by shell like command
                        for leaf_token in layer4_token.split("="):  # parse by =
                            parsed_tokens.append(leaf_token)
        return parsed_tokens
   
 
    def fit(self, input_pyspark_dataframe):
        """ call this function to vectorized a string column
        Args: 
            input_pyspark_dataframe:        the input pyspark dataframe that contains the string column
        Returns:
            output_pyspark_dataframe: the input pyspark dataframe that contains the string column
        """
        # create dictionary from input columns
        self.__create_dictionary(input_pyspark_dataframe)
 
        # create a mapping of token --> key
        map_col = create_map([lit(x) for i in self.dictionary.items() for x in i])
       
        # vectorize the parse column using bow dictionary
        df_vectorized = self.pyspark_dataframe_with_parsed_col.withColumn(self.outputCol, transform(self.parsed_column_name, lambda x: map_col[x]))
 
        # return original pyspark_dataframe with the added column
        output_pyspark_dataframe = input_pyspark_dataframe.join(df_vectorized, on=self.uuid, how="left")
 
        return output_pyspark_dataframe
 
    def __create_dictionary(self, input_pyspark_dataframe):
        """ private function to create a bag of words from a string column
        Args: 
            input_pyspark_dataframe:        the input pyspark dataframe that contains the string column
        Returns:
            dictionary:                     a python dictionary of Bag of Words
        """
        # parse the inputString into tokens
        df_parsed= input_pyspark_dataframe.select(self.inputCol, self.uuid).dropna(subset=[self.inputCol]).withColumn(self.parsed_column_name, self.parser(col(self.inputCol)))
        self.pyspark_dataframe_with_parsed_col = df_parsed
 
        # get all unique tokens in an array
        df_array_bow = df_parsed.agg(array_distinct(flatten(collect_list(self.parsed_column_name))).alias("bow"))
 
        # explode the array into a col and assign ordered key to them
        w=Window.orderBy(lit(1))
        df_dictionary = df_array_bow.select(explode(col("bow")).alias("token")).withColumn("key",row_number().over(w))
 
        # set the dictionary
        self.dictionary  = {row['token']:row['key'] for row in df_dictionary.collect()}
       
    def get_dictionary(self):
        """ return the python dictionary of bag of words
        Args: 
            n/a
        Returns:
            self.dictionary
        """
        return self.dictionary
   
    def get_pyspark_dataframe_with_parsed_col(self):
        """ return the pyspark dataframe with the parsed column using the bag of words dictionary
        Args: 
            n/a
        Returns:
            self.pyspark_dataframe_with_parsed_col
        """
        return self.pyspark_dataframe_with_parsed_col