'''
@file: cosine_similarity.py
@author: Mir Ahmed
@details: This program queries and abstracts from files
and calculates cosine similarity scores for each of the
queries against all the abstracts
@date Created: 02/25/2018
'''


from clean_sentence import process_sentence
import math


def fetch_queries_and_words(filename):
    '''
    reads data from the file and returns a dictionary
    consists of keys that represent query ids. The value
    is a list that consists of words in the query. The
    function also returns a set which contains all the unique
    words read from the file.
    '''

    query_id = 1
    queries = {}
    all_words_in_queries = set()
    with open(filename, 'r') as file:
        data = file.readlines()
        sentence = ""
        index = 0
        while index != len(data):
            if data[index].startswith('.W'):
                sentence_begin_index = index + 1
                index = sentence_begin_index
                sentence = ""
                while not data[sentence_begin_index].startswith('.I'):
                    temp = data[sentence_begin_index].rstrip('\n')
                    sentence += temp + " "
                    sentence_begin_index += 1
                    index += 1
                    if index == len(data):
                        break
                # we have the sentences here for an abstract now.
                token_list = process_sentence(sentence)
                queries[query_id] = token_list
                query_id += 1
                # add the words in the set
                for token in token_list:
                    if token not in all_words_in_queries:
                        all_words_in_queries.add(token)
            else:
                index += 1
    file.close()
    return queries, all_words_in_queries


def fetch_abstract_and_words(filename):
    '''
        reads data from the file and returns a dictionary
        consists of keys that represent abstract ids. The value
        is a list that consists of words in the abstract. The
        function also returns a set which contains all the unique
        words read from the file.
        '''

    abstract_id = 1
    abstracts = {}
    words_in_all_abstracts = set()
    with open(filename, 'r') as file:
        data = file.readlines()
        sentence = ""
        index = 0
        while index != len(data):
            if data[index].startswith('.W'):
                sentence_begin_index = index+1
                index = sentence_begin_index
                sentence = ""
                while not data[sentence_begin_index].startswith('.I'):
                    temp = data[sentence_begin_index].rstrip('\n')
                    sentence += temp + " "
                    sentence_begin_index += 1
                    index += 1
                    if index == len(data):
                        break
                # we have the sentences here for an abstract now.
                token_list = process_sentence(sentence)
                abstracts[abstract_id] = token_list
                abstract_id += 1
                # add the words in the set
                for token in token_list:
                    if token not in words_in_all_abstracts:
                        words_in_all_abstracts.add(token)
            else:
                index += 1
    file.close()
    return abstracts, words_in_all_abstracts


def calculate_term_frequencies(docs, all_words):
    '''
        Provided a copus (a collection of documents), in our case the "docs" is
        the dictionary returned by the fetch_queries_and_words function. Using the
        dictionary, it calculates the term frequencies for each of the words in
        the abstract.
    '''
    tf_dict = {}
    for id in docs:
        token_list_for_a_doc = docs[id]
        term_freq_for_a_doc = dict.fromkeys(all_words, 0)
        for token in token_list_for_a_doc:
            term_freq_for_a_doc[token] += 1
        tf_dict[id] = term_freq_for_a_doc
    return tf_dict


def calculate_inverse_document_frequency(docs, all_words):
    '''
    Provided a copus (a collection of documents), in our case the "docs" is
    the dictionary returned by the fetch_abstract_and_words function. Using the
    dictionary, it calculates the inverse document frequencies for each of the words in
    the abstract.
    '''

    number_of_documents = len(docs)
    idf_dict = {}
    for id in docs:
        token_list_for_a_doc = docs[id]
        idf_freq_for_a_doc = dict.fromkeys(all_words, 0)
        for token in token_list_for_a_doc:
            number_of_documents_containing_token = get_document_count_for_idf(token, docs)
            idf_freq_for_a_doc[token] = math.log(number_of_documents/float(number_of_documents_containing_token))
        idf_dict[id] = idf_freq_for_a_doc
        print('Finished: document', id)
    return idf_dict


def get_document_count_for_idf(word, docs):
    '''
    given a word and a collection of documents, this functions returns
    the number of documents that contain the word "word".
    '''

    count = 0
    for id in docs:
        token_list_for_a_doc = docs[id]
        if word in token_list_for_a_doc:
            count += 1
    return count


def calculate_tf_idf(tf, idf, all_words):
    '''
    given a tf (a term freqency dictionary returned by function above) and
    a idf (inverse document freqency dictionary dictionary returned by function above),
    this function calculates the tf_idf score for each of the words read from the
    dictionaries, stores it in another dictionary and returns it.
    '''
    tfidf_dict = {}
    for id in tf:
        token_list_for_a_doc = tf[id]
        tf_idf_for_a_doc = dict.fromkeys(all_words, 0)
        for token in token_list_for_a_doc:
            # optimization: if either tf or idf is 0, then stop looking up on the table
            # since tf*idf = 0
            if tf[id][token] == 0 or idf[id][token] == 0:
                tf_idf_for_a_doc[token] = 0
            else:
                tf_idf = tf[id][token] * idf[id][token]
                tf_idf_for_a_doc[token] = tf_idf
        tfidf_dict[id] = tf_idf_for_a_doc
    return tfidf_dict


def print_dict(dict):
    '''
    helper function to print the keys and
    values of a dictionary provided as a param to
    the function
    '''
    for key in dict:
        print(key, '-->', dict[key])


def get_count_non_zero_tfidf_for_a_doc(doc_id, tfidf):
    '''
    helper function to get the count of non-zero tf_idf values
    in a document
    '''
    count = 0
    tfidf_dict = tfidf[doc_id]
    for word in tfidf_dict:
        if tfidf_dict[word] != 0:
            count += 1
    return count


def generate_vectors_for_cosine_similarity(queries, abstracts, query_tfidf, abstract_tfidf):
    '''
    genrates vectors to calculate cosine similarities and stores the result in a dictionary
    and returns it. Given a query id in the returned dictionary, it returns another
    dictionary that has keys which represents abstract ids and values as a list of
    tf_idfs for that abstract.
    '''
    vectors_for_cosine_similarities = {}
    for query_id in queries:
        token_list_for_a_query = queries[query_id]
        query_format = "query_" + str(query_id) + "_vector"
        temp = {}
        tfidf_list_for_a_query = []
        for word in token_list_for_a_query:
            tfidf_list_for_a_query.append(query_tfidf[query_id][word])
        temp[query_format] = tfidf_list_for_a_query
        for abstract_id in abstracts:
            abstract_format = "abstract_" + str(abstract_id) + "_vector"
            temp[abstract_format] = []
            token_list_for_an_abstract = abstracts[abstract_id]
            for word in token_list_for_a_query:
                if word in token_list_for_an_abstract:
                    temp[abstract_format].append(abstract_tfidf[abstract_id][word])
                else:
                    temp[abstract_format].append(0)
        vectors_for_cosine_similarities[query_id] = temp
    return vectors_for_cosine_similarities


def calculate_consine_similarities(vectors):
    '''
    returns a dictionaries that contains cosine similarities for each of the
    queries against each of the abstracts. dictionary format: the returned dictionary
    contains keys which represent query ids. For a query id provided in the dictionary,
    it returns another dictionary that has abstract ids as the keys and cosine scores
    as values.
    '''
    cosine_similarities = {}
    for query_id in vectors:
        vectors_for_a_query = vectors[query_id] # a dictionary with keys as vector numbers
        query_key_format = "query_" + str(query_id) + "_vector"
        a_query_vector = vectors_for_a_query[query_key_format]
        del vectors_for_a_query[query_key_format]
        result = {}
        for a_formatted_abstract_vector_id in vectors_for_a_query:
            actual_abstract_id = a_formatted_abstract_vector_id.split('_')[1]
            an_abstract_vector = vectors_for_a_query[a_formatted_abstract_vector_id] # returns a list
            numerator = 0
            denominator_1 = 0
            denominator_2 = 0
            cosine_similarity = 0
            for i in range(len(a_query_vector)):
                numerator += a_query_vector[i] * an_abstract_vector[i]
                denominator_1 += a_query_vector[i]**2
                denominator_2 += an_abstract_vector[i]**2
            try:
                cosine_similarity = float(numerator) / float(math.sqrt(denominator_1 * denominator_2))
            except ZeroDivisionError:
                cosine_similarity = 0
            key_format = str(query_id) + '_' + str(actual_abstract_id) + '_similarity'
            # if cosine_similarity > 4:
            #     print('greater than 4', cosine_similarity)
            # elif cosine_similarity > 3:
            #     print('greater than 3', cosine_similarity)
            # elif cosine_similarity > 2:
            #     print('greater than 2', cosine_similarity)
            result[key_format] = cosine_similarity
        cosine_similarities[query_id] = result
        print("cosine similarity for " + str(query_id) + ' finished')
    return cosine_similarities


def write_to_file(cosine_similarities, filename):
    '''
    writes the result taken from the cosine similarity dictionary into a file
    where each line is seperated by space containing three values. The first value
    is the query id, followed by the abstract id and cosine similarity score of the
    query id against the abstract id.
    '''
    query_ids_sorted = list(cosine_similarities.keys())
    query_ids_sorted.sort()
    with open(filename, 'w') as out:
        for id in query_ids_sorted:
            similarities_for_an_id = cosine_similarities[id]  # a dictionary of similarities for one query
            sorted_scores = [v[0] for v in sorted(similarities_for_an_id.items(), key=lambda kv: (-kv[1], kv[0]))] # returns the keys sorted by values descending.
            for key in sorted_scores:
                abstract_id = key.split('_')[1]
                output_line = str(id) + '\t' + str(abstract_id) + '\t' + str(similarities_for_an_id[key]) + '\n'
                out.write(output_line)


if __name__ == "__main__":
    queries, words1 = fetch_queries_and_words('cran.qry')
    abstracts, words2 = fetch_abstract_and_words('cran.all.1400')
    qtf = calculate_term_frequencies(queries, words1)
    atf = calculate_term_frequencies(abstracts, words2)
    qidf = calculate_inverse_document_frequency(queries, words1)
    aidf = calculate_inverse_document_frequency(abstracts, words2)
    tf_idf_queries = calculate_tf_idf(qtf, qidf, words1)
    tf_idf_abstracts = calculate_tf_idf(atf, aidf, words2)
    vectors = generate_vectors_for_cosine_similarity(queries, abstracts, tf_idf_queries, tf_idf_abstracts)
    cosine_similarities = calculate_consine_similarities(vectors)
    write_to_file(cosine_similarities, 'ma3599_output.txt')