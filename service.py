import cherrypy
import random
import argparse
import logging
import librosa
import numpy as np
import tensorflow as tf


class HTTPService:
    
    def __init__(self, sess, args):
        self.sess = sess
        self.args = args

    def start(self):
        config = {
            "global": {
                "server.socket_port": 8765,
                 "server.socket_host": "0.0.0.0" 
             }
        }
        cherrypy.quickstart(self, "/", config)

    @cherrypy.expose
    def index(self):
        return '''
            <html>
            <form action="upload" enctype="multipart/form-data"
                  method="post">
                Upload music: <input music="music" type="file"></input>
                <input type="submit"></input>
            </form>
            </html>'''

    @cherrypy.expose
    def upload(self, music):
        data = music.file.read()
        with open(".downloadmusic", "wb") as out_file:
            out_file.write(data)
        return "File size: %d" % len(data)


def parse_args():
    parser = argparse.ArgumentParser(description='music')
    parser.add_argument('--checkpoint_path', help='checkpoint path', required=True, type=str)
    args = parser.parse_args()
    return args

    
def main(args):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(args.checkpoint_path + ".meta")
        saver.restore(sess, args.checkpoint_path)
        service = HTTPService(sess, args)
        service.start()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    main(parse_args())
