import tweepy 
import numpy as np
import pickle
import streamlit as st
import  re
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
import string
str_punc = string.punctuation

lemmatizer = WordNetLemmatizer()
url_regex = r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"""

def auth_user():
    try:
        global api
        consumer_key = 'twitter_consumer_key'
        consumer_secret = 'twitter_consumer_secret'

        access_token = 'twitter_access_token'
        access_token_secret = 'twitter_access_token_secret'


        auth=tweepy.OAuthHandler(consumer_key,consumer_secret)
        auth.set_access_token(access_token,access_token_secret)
        api=tweepy.API(auth)    

        print('Authentication success.')
    
    except:
        print('Authentication Failed')


def get_latest_tweets(username,n):
    #try:
        tweets=tweepy.Cursor(api.user_timeline, screen_name=username,include_rts=False ,tweet_mode="extended",exclude_replies=True).items(n)
        
        tweets_list=[tweet.full_text for tweet in tweets]

        return tweets_list

    #except:
        print('Could not find user, try with different username.')


def preprocess(tweets):
    '''
    input -

    output -> preprocessed and vectorized array

    Preprocesses raw text, vectorizes it so that it is ready for the model to make predicitons.

    ''' 
    captions=[]
    for tweet in tweets:
        text = tweet.lower()
       
        text = re.sub('http.*?([ ]|\|\|\||$)', ' ', text)
   
        text = re.sub(url_regex, ' ', text)
       
        text = re.sub('['+re.escape(str_punc)+']'," ",  text)
   
        text = re.sub('(\[|\()*\d+(\]|\))*', ' ', text)
   
        # Remove string marks
        text = re.sub('[’‘“\.”…–]', '', text)
        text = re.sub('[^(\w|\s)]', '', text)
        text = re.sub('(gt|lt)', '', text)
        
        captions.append(text)
        
    processed_captions=[]
    #print(captions)
    for caption in captions:
        tokens=nltk.word_tokenize(caption)
        #print('='*50,'TOKENIZATION')
        #print(tokens)

        lem = map(lemmatizer.lemmatize, tokens)

        #lemma=[lemmatizer.lemmatize(word) for word in tokens]
        res=' '.join(lem)
        #print('='*50,'LEMMATIZED TEXT')
        #print(res)
        processed_captions.append(res)
        
    with open('Tfidf.pk','rb') as fin:
        tfidf=pickle.load(fin)
        
    final_data=tfidf.transform(processed_captions)
    return final_data
    

def make_prediciton(final_data):
    '''
    input -> raw caption

    output -> INFP/ENTJ ETC ETC


    '''
    loaded_model = pickle.load(open('model', 'rb'))
    result=loaded_model.predict(final_data)
    
    labels=['ENFJ','ENFP','ENFP','ENTP','ESFJ','ESTJ','ESTP','INFJ','INFP','INTP','ISFJ','ISFP','ISTJ','ISTP']
    return labels[np.argmax(result)]

    
    # preprocess first

    # model prediction     

    # return output 


if __name__=='__main__':
    auth_user()

    username='@_saumya__'
    
    captions_list=get_latest_tweets(username,200)
    st.header('SHOWING RESULTS FOR : {}'.format(username))
    st.subheader('Showing 5 most recent tweets:')
    st.write(captions_list[:5])
    
    preprocessed_captions=preprocess(captions_list)
    result=make_prediciton(preprocessed_captions)
    
    st.write('Model predicts the personality to be a : ',result)