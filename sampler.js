export class Sampler {
    constructor(chunks) {
        this.chunks = chunks;
        this.total_size = chunks[0].length;
        this.boundaries = [0];
        this.boundaries.push(this.total_size);
    }

    sample(length) {
        while (true) {
            var index = this.getRandomInt(this.total_size - length - 1);
            if (this.boundaries[1] > index + length) {
                var within_chunk = index;
                return this.chunks[0].slice(within_chunk, within_chunk + length);
            }
        }
    }

    getRandomInt(max) {
        return Math.floor(Math.random() * Math.floor(max));
    }

    binarySearch(index, lo, hi) {
        if ((this.boundaries[lo] > index) || this.boundaries[hi] <= index) {
            return null;
        }
        while (hi > lo + 1) {
            var mid = Math.floor((lo + hi) / 2)
            if (this.boundaries[mid] > index) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        return hi;
    }

    sampleBatch(batchSize) {
        var batchArr = []
        for (var i = 0; i < batchSize; i++) {
            batchArr.push(this.sample(1024));
        }
        return batchArr;
    }


}

// module.exports = {
//     Sampler: Sampler
// }