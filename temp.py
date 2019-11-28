from nltk.tag import StanfordPOSTagger
import os
java_path = "C:/Program Files/Java/jdk1.8.0_181/bin/java.exe"
os.environ["JAVAHOME"] = java_path
stanford_dir = "C:/NLP_Programs/stanford-postagger-2018-10-16"
modelfile = stanford_dir + "/models/english-bidirectional-distsim.tagger"
jarfile = stanford_dir + "/stanford-postagger.jar"

tagger = StanfordPOSTagger(model_filename=modelfile, path_to_jar=jarfile)

print(
    tagger.tag_sents(
        sent.split() for sent in
        ["Yo im your deep learning mama", "Modi is besht", "Please work ma!"]))
x = [['11', '222'], ['33', '444']]
x = x + [['33', '444'], ['11', '222']]
print(x)