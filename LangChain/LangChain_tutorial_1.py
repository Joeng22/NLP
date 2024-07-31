'''
Lang chain CharacterTextSplitter vs RecursiveCharacterTextSplitter

CharacterTextSplitter class doesn't use the chunk_size or chunk_overlap parameters in its split_text method, 
which is why it doesn't split the text into chunks of the specified size and overlap.
if you have a long string and want to divide it into smaller chunks based on a particular character 

RecursiveCharacterTextSplitter works to reorganize the texts into chunks of the specified chunk_size, with chunk overlap where appropriate. 

'''

from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

def main():
    text = """What I Worked On

    February 2021

    Before college the two main things I worked on, outside of school, were writing and programming. I didn't write essays. I wrote what beginning writers were supposed to write then, and probably still are: short stories. My stories were awful. They had hardly any plot, just characters with strong feelings, which I imagined made them deep.

    The first programs I tried writing were on the IBM 1401 that our school district used for what was then called "data processing." This was in 9th grade, so I was 13 or 14. The school district's 1401 happened to be in the basement of our junior high school, and my friend Rich Draves and I got permission to use it. It was like a mini Bond villain's lair down there, with all these alien-looking machines — CPU, disk drives, printer, card reader — sitting up on a raised floor under bright fluorescent lights.
    """

    text_splitter_recusrive = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    docs_rec = text_splitter_recusrive.split_text(text)

    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    docs = text_splitter.split_text(text)

    print(len(docs_rec))
    for each in docs_rec:
        print(each ,"\n")

    print(len(docs))
    for each in docs:
        print(each ,"\n")

if __name__=="__main__":
    main()