/**
 * Shared Audio and Utility Functions
 */

class AudioUtils {
    /**
     * Convert Float32 audio buffer to 16-bit PCM
     * @param {Float32Array} float32Array 
     * @returns {Int16Array}
     */
    static floatTo16BitPCM(float32Array) {
        const buffer = new ArrayBuffer(float32Array.length * 2);
        const view = new DataView(buffer);
        let offset = 0;
        
        for (let i = 0; i < float32Array.length; i++, offset += 2) {
            let s = Math.max(-1, Math.min(1, float32Array[i]));
            view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        }
        
        return new Int16Array(buffer);
    }

    /**
     * Convert base64 string to ArrayBuffer
     * @param {string} base64 
     * @returns {ArrayBuffer}
     */
    static base64ToArrayBuffer(base64) {
        const binaryString = window.atob(base64);
        const len = binaryString.length;
        const bytes = new Uint8Array(len);
        for (let i = 0; i < len; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        return bytes.buffer;
    }

    /**
     * Create audio context with fallback for older browsers
     * @returns {AudioContext}
     */
    static createAudioContext() {
        if (window.AudioContext) {
            return new AudioContext();
        } else if (window.webkitAudioContext) {
            return new window.webkitAudioContext();
        } else {
            throw new Error("AudioContext not supported");
        }
    }
}
