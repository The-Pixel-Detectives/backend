from elasticsearch import Elasticsearch

# Elasticsearch connection settings
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

index_name = 'transcriptions'  # The index we will use in Elasticsearch

def index_transcription(filename: str, content: str, video_id: str, group_id: str, frame_idx: str, fps: int):
    """
    Index a transcription into Elasticsearch with additional metadata.
    :param filename: The name of the transcription file.
    :param content: The content of the transcription.
    :param video_id: The video ID associated with this transcription.
    :param group_id: The group ID associated with this transcription.
    """
    # Log the document being indexed
    # print(f"Indexing file: {filename}, video_id: {video_id}, group_id: {group_id}")

    # Indexing the transcription into Elasticsearch with metadata
    response = es.index(index=index_name, body={
        'filename': filename,
        'content': content,
        'video_id': video_id,
        'group_id': group_id,
        'frame_idx': int(frame_idx),
        'fps': fps,
    })

    # Log Elasticsearch response
    # print(f"Elasticsearch response: {response}")

def keyword_search(keyword: str):
    """
    Perform a keyword search using Elasticsearch.
    :param keyword: The search keyword.
    :return: List of filenames containing the keyword.
    """
    result = es.search(index=index_name, body={
        "query": {
            "match": {
                "content": keyword
            }
        }
    })

    # Return a list of filenames that match the keyword
    return [hit['_source']['filename'] for hit in result['hits']['hits']]


def fuzzy_search(keyword: str, fuzziness: str = "AUTO"):
    """
    Perform a fuzzy search using Elasticsearch.
    :param keyword: The keyword to search for.
    :param fuzziness: The fuzziness level (AUTO, 1, 2, etc.).
    :return: List of filenames containing fuzzy matches.
    """
    result = es.search(index=index_name, body={
        "query": {
            "match": {
                "content": {
                    "query": keyword,
                    "fuzziness": fuzziness
                }
            }
        }
    })

    # Return a list of filenames that match the fuzzy search
    return [hit['_source']['filename'] for hit in result['hits']['hits']]
