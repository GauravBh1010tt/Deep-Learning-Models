def prepare_data(sent,vocab_size=2000):
    
    START = '$_START_$'
    END = '$_END_$'
    unk_token = '$_UNK_$'  
    sentence = ["%s %s %s" % (START,x,END) for x in sent]
    tokenize_sent = [regexp_tokenize(x, 
                                     pattern = '\w+|$[\d\.]+|\S+') for x in sentence]
    
    freq = nltk.FreqDist(itertools.chain(*tokenize_sent))
    print 'found ',len(freq),' unique words'
    vocab = freq.most_common(vocab_size - 1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unk_token)
    
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
    
    for i,sent in enumerate(tokenize_sent):
        tokenize_sent[i] = [w if w in word_to_index else unk_token for w in sent]
    #X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenize_sent])
    X_train = []
    for i in tokenize_sent:
        temp = []
        for j in i[:-1]:
            temp.append(word_to_index[j])
        X_train.append(temp)    
    X_train = np.asarray(X_train)
    Y_train = []
    for i in tokenize_sent:
        temp = []
        for j in i[1:]:
            temp.append(word_to_index[j])
        Y_train.append(temp)
    Y_train = np.asarray(Y_train)
    
    return X_train,Y_train
