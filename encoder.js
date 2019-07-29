var fetch = require('node-fetch');
var fs = require('fs');
var LRU = require("lru-cache")

function bytes_to_unicode() {
    var bs = range("!".charCodeAt(0), "~".charCodeAt(0) + 1).concat(range("¡".charCodeAt(0), "¬".charCodeAt(0) + 1), range("®".charCodeAt(0), "ÿ".charCodeAt(0) + 1));
    var cs = range("!".charCodeAt(0), "~".charCodeAt(0) + 1).concat(range("¡".charCodeAt(0), "¬".charCodeAt(0) + 1), range("®".charCodeAt(0), "ÿ".charCodeAt(0) + 1));
    
    n = 0
    for (var b = 0; b < 256; b++) {
        if (!(bs.includes(b))){
            bs.push(b)
            cs.push(256+n)
            n += 1
        } 
    }
    new_cs = []
    for (var i = 0; i < cs.length; i++) {
        new_cs.push(String.fromCharCode(cs[i]));
    }
    var zipped = zip(bs, new_cs);
    return dict(zipped);
}
    
function range(start, finish) {
    var size = finish - start;
    return [...Array(size).keys()].map(i => i + start);
}
    
function zip(arr1, arr2) {
    return arr1.map(function(e, i) {
        return [e, arr2[i]];
    })
}

function dict(array) {
    dictionary = {};
    array.forEach(tuple => {
        dictionary[tuple[0]] = tuple[1];
    })
    return dictionary
}

function swap(json){
    var ret = {};
    for(var key in json){
      ret[json[key]] = key;
    }
    return ret;
}

function get_pairs(word) {
    pairs = new Set([]);
    var prev_char = word[0]
    for (var i = 1; i < word.length; i++) {
        pairs.add((prev_char, word[i]));
        prev_char = word[i];
    }
    return pairs
}

function cache_get(cache, key) {
    var value = cache.get(key);
    if (typeof(value) === undefined) {
        value = this.bytes_encoder[key]
        this.cache.set(key, value)
    }
    return value
}

class Encoder {
    constructor(encoder, bpe_merges) {
        this.encoder = encoder;
        //console.log(encoder)
        this.decoder = swap(encoder);
        this.errors = 'replace';
        this.bytes_encoder = bytes_to_unicode();
        this.bytes_decoder = swap(this.bytes_encoder);
        this.bytes_cache1 = new LRU(128);
        this.bytes_cache2 = new LRU(128);
        this.bpe_ranks = dict(zip(bpe_merges, range(0, bpe_merges.length)));
        this.cache = {};

        this.pat = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/g;
    }

    bpe(token) {
        if (token in this.cache) {
            return this.cache[token];
        }

        var word = token.split("");
        var pairs = get_pairs(word)

        if (pairs.size == 0) {
            return token;
        }

        var bpe_ranks = this.bpe_ranks
        pairs = Array.from(pairs);
        while (true) {
            var bigram = pairs.filter(function(pair) {
                return pair in bpe_ranks;
            }).map(function (pair) {
                return (pair, bpe_ranks[pair])
            });

            if (bigram.length == 0) {
                break;
            } 
            bigram = bigram.reduce(function (pair1, pair2) {
                return (pair1[1] < pair2[1]) ? pair1[0] : pair2[0];
            });

            if (!(bigram in bpe_ranks)) {
                break;
            }

            var first = bigram[0];
            var second = bigram[1];

            new_word = [];
            var i = 0;
            while (i < word.length) {
                var j = word.indexOf(first);
                if (j == -1) {
                    new_word.concat(word.substring(i).split(""));
                    break;
                } else {
                    new_word.concat(word.substring(i, j).split(""));
                    i = j;
                }

                if (word.charAt(i) == first && i < word.length - 1 && word.charAt(i + 1) == second) {
                    new_word.push(first.concat(second));
                    i += 2;
                } else {
                    new_word.push(word.charAt(i));
                    i += 1;
                }
            }

            word = new_word;
            if (word.length == 1) {
                break;
            } else {
                pairs = get_pairs(word);
            }
        }

        word = word.join(" ");
        this.cache[token] = word;
        return word;
    }  

    encode(text) {
        var bpe_tokens = [];
        var cache = this.bytes_cache1;
        //console.log(bytes_encoder)
        var encoder = this.encoder;
        var num = 0;
        text.match(this.pat).forEach(element => {
            if (num % 10000 == 0) {
                console.log(num)
            }
            var token = this.encode_utf8(element).split().map(function(char) {
                return cache_get(cache, char.charCodeAt(0));
            }).join('');
            bpe_tokens = bpe_tokens.concat(this.bpe(token).split(' ').map(element => {
                return encoder[element];
            }));
            num += 1;
        });
        console.log("BPE TOKENS", bpe_tokens.length)
        return bpe_tokens;
    }

    encode_utf8(s) {
        return unescape(encodeURIComponent(s));
    }
      
    decode_utf8(s) {
        return decodeURIComponent(escape(s));
    }

    decode(tokens) {
        var cache = this.cache2_get;
        var decoder = this.decoder
        text = tokens.map(element => {
            return decoder[String.fromCharCode(element)];
        }).join('');

        text = text.split('').map(element => {
            return cache(c);
        })
        
        return this.decode_utf8(text);
    }

    load_dataset(text) {
        var token_chunks = [];
        if (text.length <= 50000) {
            text = text.concat('<|endoftext|>');
        }
        token_chunks.push(this.encode(text));
        return token_chunks;
    }
}

function get_encoder() {
    var encoder_json = JSON.parse(fs.readFileSync('encoder.json', 'utf8'));
    var vocab_bpe = fs.readFileSync('vocab.bpe', 'utf8');
    var bpe_merges = [];
    var temp_arr = vocab_bpe.split('\n')
    temp_arr = temp_arr.slice(1, temp_arr.length - 1);
    for (var i = 0; i < temp_arr.length; i++) {
        bpe_merges.push(temp_arr[i].split(' '))
    }
    return new Encoder(encoder_json, bpe_merges);
}

module.exports = {
    get_encoder: get_encoder,
    Encoder: Encoder
};

    