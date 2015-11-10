package water.rapids;

import water.*;
import water.fvec.Chunk;
import water.fvec.Frame;
import water.fvec.Vec;
import water.util.ArrayUtils;
import water.util.Log;
import water.util.Pair;
import water.util.PrettyPrint;

import java.util.Arrays;
import java.util.Hashtable;

class RadixCount extends MRTask<RadixCount> {
  public static class Long2DArray extends Iced {
    Long2DArray(int len) { _val = new long[len][]; }
    long _val[][];
  }
  Long2DArray _counts;
  int _biggestBit;
  int _col;
  boolean _isLeft; // used to determine the unique DKV names since DF._key is null now and before only an RTMP name anyway

  RadixCount(boolean isLeft, int biggestBit, int col) {
    _isLeft = isLeft;
    _biggestBit = biggestBit;
    _col = col;
  }

  // make a unique deterministic key as a function of frame, column and node
  // make it homed to the owning node
  static Key getKey(boolean isLeft, int col, H2ONode node) {
    Key ans = Key.make("__radix_order__MSBNodeCounts_col" + col + "_node" + node.index() + (isLeft ? "_LEFT" : "_RIGHT"));
    // Each node's contents is different so the node number needs to be in the key
    // TO DO: need the biggestBit in here too, that the MSB is offset from
    //Log.info(ans.toString());
    return ans;
  }

  @Override protected void setupLocal() {
    _counts = new Long2DArray(_fr.anyVec().nChunks());
  }

  @Override public void map( Chunk chk ) {
    long tmp[] = _counts._val[chk.cidx()] = new long[256];
    int shift = _biggestBit-8;
    if (shift<0) shift = 0;
    // TO DO: assert chk instanceof integer or enum;  -- but how?  // alternatively: chk.getClass().equals(C8Chunk.class)
    for (int r=0; r<chk._len; r++) {
      tmp[(int) (chk.at8(r) >> shift & 0xFFL)]++;  // forget the L => wrong answer with no type warning from IntelliJ
      // TO DO - use _mem directly. Hist the compressed bytes and then shift the histogram afterwards when reducing.
    }
  }

  @Override protected void closeLocal() {
    //Log.info("Putting MSB counts for column " + _col + " over my chunks (node " + H2O.SELF + ") for frame " + _frameKey);
    //Log.info("Putting");
    DKV.put(getKey(_isLeft, _col, H2O.SELF), _counts);
    // just the MSB counts per chunk on this node.  Most of this spine will be empty here.  TO DO: could condense to just the chunks on this node but for now, leave sparse.
    // We'll use this sparse spine right now on this node and the reduce happens on _o and _x later
    //super.postLocal();
  }
}

class SplitByMSBLocal extends MRTask<SplitByMSBLocal> {
  private transient long _counts[][];
  transient long _o[][][];  // transient ok because there is no reduce here between nodes, and important to save shipping back to caller.
  transient byte _x[][][];
  boolean _isLeft;
  int _biggestBit, _batchSize, _bytesUsed[], _keySize;
  int[]_col;
  Key _linkTwoMRTask;

  static Hashtable<Key,SplitByMSBLocal> MOVESHASH = new Hashtable<>();
  SplitByMSBLocal(boolean isLeft, int biggestBit, int keySize, int batchSize, int bytesUsed[], int[] col, Key linkTwoMRTask) {
    _isLeft = isLeft;
    _biggestBit = biggestBit; _batchSize=batchSize; _bytesUsed = bytesUsed; _col = col;
    _keySize = keySize;  // ArrayUtils.sum(_bytesUsed) -1;
    _linkTwoMRTask = linkTwoMRTask;
    //setProfile(true);
  }

  @Override protected void setupLocal() {
    // First cumulate MSB count histograms across the chunks in this node
    // Log.info("Getting RadixCounts for column " + _col[0] + " from myself (node " + H2O.SELF + ") for Frame " + _frameKey );
    // Log.info("Getting");
    _counts = ((RadixCount.Long2DArray)DKV.getGet(RadixCount.getKey(_isLeft, _col[0], H2O.SELF)))._val;   // get the sparse spine for this node, created and DKV-put above
    // try {
    //   Thread.sleep(10000);
    // } catch (InterruptedException e) {
    //   e.printStackTrace();
    // }
    long _MSBhist[] = new long[256];  // total across nodes but retain original spine as we need that below
    int nc = _fr.anyVec().nChunks();
    for (int c = 0; c < nc; c++) {
      if (_counts[c]!=null) {
        for (int h = 0; h < 256; h++) {
          _MSBhist[h] += _counts[c][h];
        }
      }
    }
    if (ArrayUtils.maxValue(_MSBhist) > Math.max(1000, _fr.numRows() / 20 / H2O.CLOUD.size())) {  // TO DO: better test of a good even split
      Log.warn("RadixOrder(): load balancing on this node not optimal (max value should be <= "
              + (Math.max(1000, _fr.numRows() / 20 / H2O.CLOUD.size()))
              + " " + Arrays.toString(_MSBhist) + ")");
    }
    /*System.out.println("_MSBhist on this node (biggestBit==" + _biggestBit + ") ...");
    for (int m=1; m<16; m+=2) {
      // print MSB 0-5, 16-21, etc to get a feel for distribution without wrapping line width when large numbers
      System.out.print(String.format("%3d",m*16) + ": ");
      for (int n=0; n<6; n++) System.out.print(String.format("%9d",_MSBhist[m*16 + n]) + " ");
      System.out.println(" ...");
    }*/
    // shared between threads on the same node, all mappers write into distinct locations (no conflicts, no need to atomic updates, etc.)
    System.out.print("Allocating _o and _x buckets on this node with known size up front ... ");
    long t0 = System.nanoTime();
    _o = new long[256][][];
    _x = new byte[256][][];  // for each bucket, there might be > 2^31 bytes, so an extra dimension for that
    for (int msb = 0; msb < 256; msb++) {
      if (_MSBhist[msb] == 0) continue;
      int nbatch = (int) (_MSBhist[msb]-1)/_batchSize +1;  // at least one batch
      int lastSize = (int) (_MSBhist[msb] - (nbatch-1) * _batchSize);   // the size of the last batch (could be batchSize)
      assert nbatch == 1;  // Prevent large testing for now.  TO DO: test nbatch>0 by reducing batchSize very small and comparing results with non-batched
      assert lastSize > 0;
      _o[msb] = new long[nbatch][];
      _x[msb] = new byte[nbatch][];
      int b;
      for (b = 0; b < nbatch-1; b++) {
        _o[msb][b] = new long[_batchSize];          // TO DO?: use MemoryManager.malloc8()
        _x[msb][b] = new byte[_batchSize * _keySize];
      }
      _o[msb][b] = new long[lastSize];
      _x[msb][b] = new byte[lastSize * _keySize];
    }
    System.out.println("done in " + (System.nanoTime() - t0 / 1e9));

    // TO DO: otherwise, expand width. Once too wide (and interestingly large width may not be a problem since small buckets won't impact cache),
    // start rolling up bins (maybe into pairs or even quads)
    // System.out.print("Columwise cumulate of chunk level MSB hists ... ");
    for (int msb = 0; msb < 256; msb++) {
      long rollSum = 0;  // each of the 256 columns starts at 0 for the 0th chunk. This 0 offsets into x[MSBvalue][batch div][mod] and o[MSBvalue][batch div][mod]
      for (int c = 0; c < nc; c++) {
        if (_counts[c] == null) continue;
        long tmp = _counts[c][msb];
        _counts[c][msb] = rollSum; //Warning: modify the POJO DKV cache, but that's fine since this node won't ask for the original DKV.get() version again
        rollSum += tmp;
      }
    }

    MOVESHASH.put(_linkTwoMRTask, this);

    // System.out.println("done");
    // NB: no radix skipping in this version (unlike data.table we'll use biggestBit and assume further bits are used).
  }


  @Override public void map(Chunk chk[]) {
    // System.out.println("Starting MoveByFirstByte.map() for chunk " + chk[0].cidx());
    long myCounts[] = _counts[chk[0].cidx()]; //cumulative offsets into o and x
    if (myCounts == null) {
      System.out.println("myCounts empty for chunk " + chk[0].cidx());
      return;  // TODO delete ... || _o==null || _x==null) return;
    }

    //int leftAlign = (8-(_biggestBit % 8)) % 8;   // only the first column is left aligned, currently. But they all could be for better splitting.

    // Loop through this chunk and write the byte key and the source row into the local MSB buckets
    long t0 = System.nanoTime();
    for (int r=0; r<chk[0]._len; r++) {    // tight, branch free and cache efficient (surprisingly)
      long thisx = chk[0].at8(r);  // - _colMin[0]) << leftAlign;  (don't subtract colMin because it unlikely helps compression and makes joining 2 compressed keys more difficult and expensive).
      int shift = _biggestBit-8;
      if (shift<0) shift = 0;
      int MSBvalue = (int) (thisx >> shift & 0xFFL);
      long target = myCounts[MSBvalue]++;
      int batch = (int) (target / _batchSize);
      int offset = (int) (target % _batchSize);
      if (_o[MSBvalue] == null) throw new RuntimeException("Internal error: o_[MSBvalue] is null. Should never happen.");
      _o[MSBvalue][batch][offset] = (long) r + chk[0].start();    // move i and the index.

      byte this_x[] = _x[MSBvalue][batch];
      offset *= _keySize; //can't overflow because batchsize was chosen above to be maxByteSize/max(keysize,8)
      for (int i = _bytesUsed[0] - 1; i >= 0; i--) {   // a loop because I don't believe System.arraycopy() can copy parts of (byte[])long to byte[]
        this_x[offset + i] = (byte) (thisx & 0xFF);
        thisx >>= 8;
      }
      for (int c=1; c<chk.length; c++) {  // TO DO: left align subsequent
        offset += _bytesUsed[c-1]-1;  // advance offset by the previous field width
        thisx = chk[c].at8(r);        // TO DO : compress with a scale factor such as dates stored as ms since epoch / 3600000L
        for (int i = _bytesUsed[c] - 1; i >= 0; i--) {
          this_x[offset + i] = (byte) (thisx & 0xFF);
          thisx >>= 8;
        }
      }
    }
    // System.out.println(System.currentTimeMillis() + " MoveByFirstByte.map() into MSB buckets for chunk " + chk[0].cidx() + " took : " + (System.nanoTime() - t0) / 1e9);
  }

  static H2ONode ownerOfMSB(int MSBvalue) {
    // TO DO: this isn't properly working for efficiency. This value should pick the value of where it is, somehow.
    //        Why not getSortedOXHeader(MSBvalue).home_node() ?
    //int blocksize = (int) Math.ceil(256. / H2O.CLOUD.size());
    //H2ONode node = H2O.CLOUD._memary[MSBvalue / blocksize];
    H2ONode node = H2O.CLOUD._memary[MSBvalue % H2O.CLOUD.size()];   // spread it around more.
    return node;
  }

  public static Key getNodeOXbatchKey(boolean isLeft, int MSBvalue, int node, int batch) {
    Key ans = Key.make("__radix_order__NodeOXbatch_MSB" + MSBvalue + "_node" + node + "_batch" + batch + (isLeft ? "_LEFT" : "_RIGHT"),
            (byte) 1, Key.HIDDEN_USER_KEY, false, SplitByMSBLocal.ownerOfMSB(MSBvalue));
    return ans;
  }

  public static Key getSortedOXbatchKey(boolean isLeft, int MSBvalue, int batch) {
    Key ans = Key.make("__radix_order__SortedOXbatch_MSB" + MSBvalue + "_batch" + batch + (isLeft ? "_LEFT" : "_RIGHT"),
            (byte) 1, Key.HIDDEN_USER_KEY, false, SplitByMSBLocal.ownerOfMSB(MSBvalue));
    return ans;
  }


  public static class OXbatch extends Iced {
    OXbatch(long[] o, byte[] x) { _o = o; _x = x; }
    long[/*batchSize or lastSize*/] _o;
    byte[/*batchSize or lastSize*/] _x;
  }

  public static Key getMSBNodeHeaderKey(boolean isLeft, int MSBvalue, int node) {
    Key ans = Key.make("__radix_order__OXNodeHeader_MSB" + MSBvalue + "_node" + node + (isLeft ? "_LEFT" : "_RIGHT"),
            (byte) 1, Key.HIDDEN_USER_KEY, false, SplitByMSBLocal.ownerOfMSB(MSBvalue));
    return ans;
  }

  public static class MSBNodeHeader extends Iced {
    MSBNodeHeader(int MSBnodeChunkCounts[/*chunks*/]) { _MSBnodeChunkCounts = MSBnodeChunkCounts;}
    int _MSBnodeChunkCounts[];   // a vector of the number of contributions from each chunk.  Since each chunk is length int, this must less than that, so int
  }

  // Push o/x in chunks to owning nodes
  void SendSplitMSB() {
    long t0 = System.nanoTime();
    Futures fs = new Futures();
    // System.out.println(System.currentTimeMillis() + " Starting MoveByFirstByte.PushFirstByteBatches() ... ");
    int nc = _fr.anyVec().nChunks();
    long forLoopTime = 0, DKVputTime = 0, sumSize = 0;

    // The map() above ran above for each chunk on this node. Although this data was written to _o and _x in the order of chunk number (because we
    // calculated those offsets in order in the prior step), the chunk numbers will likely have gaps because chunks are distributed across nodes
    // not using a modulo approach but something like chunk1 on node1, chunk2 on node2, etc then modulo after that. Also, as tables undergo
    // changes as a result of user action, their distribution of chunks to nodes could change or be changed (e.g. 'Thomas' rebalance()') for various reasons.
    // When the helper node (i.e the node doing all the A's) gets the A's from this node, it must stack all this nodes' A's with the A's from the other
    // nodes in chunk order in order to maintain the original order of the A's within the global table.
    // To to do that, this node must tell the helper node where the boundaries are in _o and _x.  That's what the first for loop below does.
    // The helper node doesn't need to be sent the corresponding chunk numbers. He already knows them from the Vec header which he already has locally.
    // TODO: perhaps write to o_ and x_ in batches in the first place, and just send more and smaller objects via the DKV.  This makes the stitching much easier
    // on the helper node too, as it doesn't need to worry about batch boundaries in the source data.  Then it might be easier to parallelize that helper part.
    // The thinking was that if each chunk generated 256 objects, that would flood the DKV with keys?
    // TODO: send nChunks * 256.  Currently we do nNodes * 256.  Or avoid DKV altogether if possible.

    for (int msb =0; msb <_o.length /*256*/; ++msb) {   // TODO this can be done in parallel, surely
      // “I found my A’s (msb=0) and now I'll send them to the node doing all the A's”
      // "I'll send you a long vector of _o and _x (batched if very long) along with where the boundaries are."
      // "You don't need to know the chunk numbers of these boundaries, because you know the node of each chunk from your local Vec header"
      long t00 = System.nanoTime();
      if(_o[msb] == null) continue;
      int numChunks = 0;  // how many of the chunks on this node had some rows with this MSB
      for (int c=0; c<nc; c++) {
        if (_counts[c] != null && _counts[c][msb] > 0)
          numChunks++;
      }
      int MSBnodeChunkCounts[] = new int[numChunks];   // make dense.  And by construction (i.e. cumulative counts) these chunks contributed in order
      int j=0;
      long lastCount = 0; // _counts are cumulative at this stage so need to undo
      for (int c=0; c<nc; c++) {
        if (_counts[c] != null && _counts[c][msb] > 0) {
          MSBnodeChunkCounts[j] = (int)(_counts[c][msb] - lastCount);  // _counts is long so it can be accumulated in-place I think.  TO DO: check
          lastCount = _counts[c][msb];
          j++;
        }
      }
      forLoopTime += System.nanoTime() - t00;

      //assert _MSBhist[msb] == ArrayUtils.sum(MSBnodeChunkCounts);
      MSBNodeHeader msbh = new MSBNodeHeader(MSBnodeChunkCounts);
      //Log.info("Putting MSB node headers for Frame " + _frameKey + " for MSB " + msb);
      //Log.info("Putting msb " + msb + " on node " + H2O.SELF.index());
      DKV.put(getMSBNodeHeaderKey(_isLeft, msb, H2O.SELF.index()), msbh, fs, true);
      for (int b=0;b<_o[msb].length; ++b) {
        OXbatch ox = new OXbatch(_o[msb][b], _x[msb][b]);   // this does not copy in Java, just references
        //Log.info("Putting OX batch for Frame " + _frameKey + " for batch " + b + " for MSB " + msb);
        //Log.info("Putting");
        sumSize += _o[msb][b].length * 8 + _x[msb][b].length * _keySize + 64;
        t00 = System.nanoTime();
        DKV.put(getNodeOXbatchKey(_isLeft, msb, H2O.SELF.index(), b), ox, fs, true);   // TODO: send to particular node
        long thist = System.nanoTime() - t00;
        // System.out.println("This DKV put took : " + thist / 1e9);  // TODO:  adding fs (does it need ,fs,true?) above didn't appear to make it go in parallel. Apx 200 prints of 0.07 = 14s total
        DKVputTime += thist;
      }

    }
    System.out.println("this node took : " + (System.nanoTime() - t0) / 1e9);
    System.out.println("  Finding the chunk boundaries in split OX took " + forLoopTime / 1e9);
    System.out.println("  DKV.put " + PrettyPrint.bytes(sumSize) + " in " + DKVputTime / 1e9 + ". Overall rate: " +
                        String.format("%.3f", sumSize / (DKVputTime/1e9) / (1024*1024*1024)) + " GByte/sec  [10Gbit = 1.25GByte/sec]");

  }
}


class SendSplitMSB extends MRTask<SendSplitMSB> {
  Key _linkTwoMRTask;
  SendSplitMSB(Key linkTwoMRTask) {
    _linkTwoMRTask = linkTwoMRTask;
  }
  @Override public void setupLocal() {
    SplitByMSBLocal.MOVESHASH.get(_linkTwoMRTask).SendSplitMSB();
    SplitByMSBLocal.MOVESHASH.remove(_linkTwoMRTask);
  }
}

// It is intended that several of these SingleThreadRadixOrder run on the same node, to utilize the cores available.
// The initial MSB needs to split by num nodes * cpus per node; e.g. 256 is pretty good for 10 nodes of 32 cores. Later, use 9 bits, or a few more bits accordingly.
// Its this 256 * 4kB = 1MB that needs to be < cache per core for cache write efficiency in MoveByFirstByte(). 10 bits (1024 threads) would be 4MB which still < L2
// Since o[] and x[] are arrays here (not Vecs) it's harder to see how to parallelize inside this function. Therefore avoid that issue by using more threads in calling split.
// General principle here is that several parallel, tight, branch free loops, faster than one heavy DKV pass per row


class SingleThreadRadixOrder extends DTask<SingleThreadRadixOrder> {
  //long _nGroup[];
  int _MSBvalue;  // only needed to be able to return the number of groups back to the caller RadixOrder
  int _keySize, _batchSize;
  long _numRows;
  Frame _fr;
  boolean _isLeft;

  private transient long _o[/*batch*/][];
  private transient byte _x[/*batch*/][];
  private transient long _otmp[][];
  private transient byte _xtmp[][];

  // TEMPs
  private transient long counts[][];
  private transient byte keytmp[];
  //public long _groupSizes[][];


  // outputs ...
  // o and x are changed in-place always
  // iff _groupsToo==true then the following are allocated and returned
  //int _MSBvalue;  // Most Significant Byte value this node is working on
  //long _nGroup[/*MSBvalue*/];  // number of groups found (could easily be > BATCHLONG). Each group has its size stored in _groupSize.
  //long _groupSize[/*MSBvalue*/][/*_nGroup div BATCHLONG*/][/*mod*/];

  // Now taken out ... boolean groupsToo, long groupSize[][][], long nGroup[], int MSBvalue
  //long _len;
  //int _byte;

  SingleThreadRadixOrder(Frame fr, boolean isLeft, int batchSize, int keySize, /*long nGroup[],*/ int MSBvalue) {
    _fr = fr;
    _isLeft = isLeft;
    _batchSize = batchSize;
    _keySize = keySize;
    //_nGroup = nGroup;
    _MSBvalue = MSBvalue;
  }

  @Override protected void compute2() {
    keytmp = new byte[_keySize];
    counts = new long[_keySize][256];

    SplitByMSBLocal.MSBNodeHeader[] MSBnodeHeader = new SplitByMSBLocal.MSBNodeHeader[H2O.CLOUD.size()];
    _numRows =0;
    for (int n=0; n<H2O.CLOUD.size(); n++) {
      // Log.info("Getting MSB " + MSBvalue + " Node Header from node " + n + "/" + H2O.CLOUD.size() + " for Frame " + _fr._key);
      // Log.info("Getting");
      MSBnodeHeader[n] = DKV.getGet(SplitByMSBLocal.getMSBNodeHeaderKey(_isLeft, _MSBvalue, n));
      if (MSBnodeHeader[n]==null) continue;
      _numRows += ArrayUtils.sum(MSBnodeHeader[n]._MSBnodeChunkCounts);   // This numRows is split into nbatch batches on that node.
      // This header has the counts of each chunk (the ordered chunk numbers on that node)
    }
    if (_numRows == 0) { tryComplete(); return; }

    // Allocate final _o and _x for this MSB which is gathered together on this node from the other nodes.
    int nbatch = (int) (_numRows -1) / _batchSize +1;   // at least one batch.    TO DO:  as Arno suggested, wrap up into class for fixed width batching (to save espc overhead)
    int lastSize = (int) (_numRows - (nbatch-1) * _batchSize);   // the size of the last batch (could be batchSize, too if happens to be exact multiple of batchSize)
    _o = new long[nbatch][];
    _x = new byte[nbatch][];
    int b;
    for (b = 0; b < nbatch-1; b++) {
      _o[b] = new long[_batchSize];          // TO DO?: use MemoryManager.malloc8()
      _x[b] = new byte[_batchSize * _keySize];
    }
    _o[b] = new long[lastSize];
    _x[b] = new byte[lastSize * _keySize];

    SplitByMSBLocal.OXbatch ox[/*node*/] = new SplitByMSBLocal.OXbatch[H2O.CLOUD.size()];
    int oxBatchNum[/*node*/] = new int[H2O.CLOUD.size()];  // which batch of OX are we on from that node?  Initialized to 0.
    for (int node=0; node<H2O.CLOUD.size(); node++) {  //TO DO: why is this serial?  Relying on
      // Log.info("Getting OX MSB " + MSBvalue + " batch 0 from node " + node + "/" + H2O.CLOUD.size() + " for Frame " + _fr._key);
      // Log.info("Getting");
      Key k = SplitByMSBLocal.getNodeOXbatchKey(_isLeft, _MSBvalue, node, /*batch=*/0);
      assert k.home();
      ox[node] = DKV.getGet(k);   // get the first batch for each node for this MSB
    }
    int oxOffset[] = new int[H2O.CLOUD.size()];
    int oxChunkIdx[] = new int[H2O.CLOUD.size()];  // that node has n chunks and which of those are we currently on?

    int targetBatch = 0, targetOffset = 0, targetBatchRemaining = _batchSize;

    for (int c=0; c<_fr.anyVec().nChunks(); c++) {
      int fromNode = _fr.anyVec().chunkKey(c).home_node().index();  // each chunk in the column may be on different nodes
      // See long comment at the top of SendSplitMSB. One line from there repeated here :
      // " When the helper node (i.e. this one, now) (i.e the node doing all the A's) gets the A's from that node, it must stack all the nodes' A's
      // with the A's from the other nodes in chunk order in order to maintain the original order of the A's within the global table. "
      // TODO: We could process these in node order and or/in parallel if we cumulated the counts first to know the offsets - should be doable and high value
      if (MSBnodeHeader[fromNode] == null) continue;
      int numRowsToCopy = MSBnodeHeader[fromNode]._MSBnodeChunkCounts[oxChunkIdx[fromNode]++];   // magically this works, given the outer for loop through global chunk
      // _MSBnodeChunkCounts is a vector of the number of contributions from each Vec chunk.  Since each chunk is length int, this must less than that, so int
      // The set of data corresponding to the Vec chunk contributions is stored packed in batched vectors _o and _x.
      int sourceBatchRemaining = _batchSize - oxOffset[fromNode];    // at most batchSize remaining.  No need to actually put the number of rows left in here
      while (numRowsToCopy > 0) {   // No need for class now, as this is a bit different to the other batch copier. Two isn't too bad.
        int thisCopy = Math.min(numRowsToCopy, Math.min(sourceBatchRemaining, targetBatchRemaining));
        System.arraycopy(ox[fromNode]._o, oxOffset[fromNode],          _o[targetBatch], targetOffset,          thisCopy);
        System.arraycopy(ox[fromNode]._x, oxOffset[fromNode]*_keySize, _x[targetBatch], targetOffset*_keySize, thisCopy*_keySize);
        numRowsToCopy -= thisCopy;
        oxOffset[fromNode] += thisCopy; sourceBatchRemaining -= thisCopy;
        targetOffset += thisCopy; targetBatchRemaining -= thisCopy;
        if (sourceBatchRemaining == 0) {
          // delete and free node's OX batch. Have moved all its contents into _o and _x. Remove so as to avoid mistakenly picking it up in future and to free up memory as early as possible.
          DKV.remove(SplitByMSBLocal.getNodeOXbatchKey(_isLeft, _MSBvalue, fromNode, oxBatchNum[fromNode]));
          // fetched the next batch :
          ox[fromNode] = DKV.getGet(SplitByMSBLocal.getNodeOXbatchKey(_isLeft, _MSBvalue, fromNode, ++oxBatchNum[fromNode]));
          if (ox[fromNode] == null) {
            // if the last chunksworth fills a batchsize exactly, the getGet above will have returned null.
            // TODO: Check will Cliff that a known fetch of a non-existent key is ok e.g. won't cause a delay/block? If ok, leave as good check.
            assert oxBatchNum[fromNode]==MSBnodeHeader[fromNode]._MSBnodeChunkCounts.length;
            assert ArrayUtils.sum(MSBnodeHeader[fromNode]._MSBnodeChunkCounts) % _batchSize == 0;
          }
          oxOffset[fromNode] = 0;
          sourceBatchRemaining = _batchSize;
        }
        if (targetBatchRemaining == 0) {
          targetBatch++;
          targetOffset = 0;
          targetBatchRemaining = _batchSize;
        }
      }
    }
    // Now remove the last batch on each node from DKV.  Have used it all now (moved into _o and _x). Remove as early as possible to save memory.
    for (int node = 0; node < H2O.CLOUD.size(); node++) {
      if (MSBnodeHeader[node] == null) continue;   // no contributions from that node
      long nodeNumRows = ArrayUtils.sum(MSBnodeHeader[node]._MSBnodeChunkCounts);
      assert nodeNumRows >= 1;
      if (nodeNumRows % _batchSize == 0) {
        // already should have been removed above in 'if (sourceBatchRemaining == 0)'
        assert ox[node] == null;
      } else {
        assert ox[node] != null;
        assert oxBatchNum[node] == (nodeNumRows-1) / _batchSize;
        DKV.remove(SplitByMSBLocal.getNodeOXbatchKey(_isLeft, _MSBvalue, node, oxBatchNum[node]));
      }
    }

    // We now have _o and _x collated from all the contributing nodes, in the correct original order.
    _xtmp = new byte[_x.length][];
    _otmp = new long[_o.length][];
    assert _x.length == _o.length;  // i.e. aligned batch size between x and o (think 20 bytes keys and 8 bytes of long in o)
    for (int i=0; i<_x.length; i++) {    // Seems like no deep clone available in Java. Maybe System.arraycopy but maybe that needs target to be allocated first
      _xtmp[i] = Arrays.copyOf(_x[i], _x[i].length);
      _otmp[i] = Arrays.copyOf(_o[i], _o[i].length);
    }
    // TO DO: a way to share this working memory between threads.
    //        Just create enough for the 4 threads active at any one time.  Not 256 allocations and releases.
    //        We need o[] and x[] in full for the result. But this way we don't need full size xtmp[] and otmp[] at any single time.
    //        Currently Java will allocate and free these xtmp and otmp and maybe it does good enough job reusing heap that we don't need to explicitly optimize this reuse.
    //        Perhaps iterating this task through the largest bins first will help java reuse heap.
    //_groupSizes = new long[256][];
    //_groups = groups;
    //_nGroup = nGroup;
    //_whichGroup = whichGroup;
    //_groups[_whichGroup] = new long[(int)Math.min(MAXVECLONG, len) ];   // at most len groups (i.e. all groups are 1 row)
    assert(_o != null);
    assert(_numRows > 0);

    // The main work. Radix sort this batch ...
    run(0, _numRows, _keySize-1);  // if keySize is 6 bytes, first byte is byte 5

    // don't need to clear these now using private transient
    // _counts = null;
    // keytmp = null;
    //_nGroup = null;

    // tell the world how many batches and rows for this MSB
    OXHeader msbh = new OXHeader(_o.length, _numRows, _batchSize);
    // Log.info("Putting MSB header for Frame " + _fr._key + " for MSB " + _MSBvalue);
    // Log.info("Putting");
    Futures fs = new Futures();
    DKV.put(getSortedOXHeaderKey(_isLeft, _MSBvalue), msbh, fs, true);
    for (b=0; b<_o.length; b++) {
      SplitByMSBLocal.OXbatch tmp = new SplitByMSBLocal.OXbatch(_o[b], _x[b]);
      // Log.info("Putting OX header for Frame " + _fr._key + " for MSB " + _MSBvalue);
      // Log.info("Putting");
      DKV.put(SplitByMSBLocal.getSortedOXbatchKey(_isLeft, _MSBvalue, b), tmp, fs, true);  // the OXbatchKey's on this node will be reused for the new keys
    }
    fs.blockForPending();
    tryComplete();
  }

  public static Key getSortedOXHeaderKey(boolean isLeft, int MSBvalue) {
    // This guy has merges together data from all nodes and its data is not "from" any particular node. Therefore node number should not be in the key.
    Key ans = Key.make("__radix_order__SortedOXHeader_MSB" + MSBvalue + (isLeft ? "_LEFT" : "_RIGHT"));  // If we don't say this it's random ... (byte) 1 /*replica factor*/, (byte) 31 /*hidden user-key*/, true, H2O.SELF);
    //if (MSBvalue==73) Log.info(ans.toString());
    return ans;
  }

  public static class OXHeader extends Iced {
    OXHeader(int batches, long numRows, int batchSize) { _nBatch = batches; _numRows = numRows; _batchSize = batchSize; }
    int _nBatch;
    long _numRows;
    int _batchSize;
  }

  int keycmp(byte x[], int xi, byte y[], int yi) {
    // Same return value as strcmp in C. <0 => xi<yi
    xi *= _keySize; yi *= _keySize;
    int len = _keySize;
    while (len > 1 && x[xi] == y[yi]) { xi++; yi++; len--; }
    return ((x[xi] & 0xFF) - (y[yi] & 0xFF)); // 0xFF for getting back from -1 to 255
  }

  public void insert(long start, int len)   // only for small len so len can be type int
/*  orders both x and o by reference in-place. Fast for small vectors, low overhead.
    don't be tempted to binsearch backwards here because have to shift anyway  */
  {
    int batch0 = (int) (start / _batchSize);
    int batch1 = (int) (start+len-1) / _batchSize;
    long origstart = start;   // just for when straddle batch boundaries
    int len0 = 0;             // same
    // _nGroup[_MSBvalue]++;  // TODO: reinstate.  This is at least 1 group (if all keys in this len items are equal)
    byte _xbatch[];
    long _obatch[];
    if (batch1 != batch0) {
      // small len straddles a batch boundary. Unlikely very often since len<=200
      assert batch0 == batch1-1;
      len0 = _batchSize - (int)(start % _batchSize);
      // copy two halves to contiguous temp memory, do the below, then split it back to the two halves afterwards.
      // Straddles batches very rarely (at most once per batch) so no speed impact at all.
      _xbatch = new byte[len * _keySize];
      System.arraycopy(_xbatch, 0, _x[batch0], (int)((start % _batchSize)*_keySize), len0*_keySize);
      System.arraycopy(_xbatch, len0*_keySize, _x[batch1], 0, (len-len0)*_keySize);
      _obatch = new long[len];
      System.arraycopy(_obatch, 0, _o[batch0], (int)(start % _batchSize), len0);
      System.arraycopy(_obatch, len0, _o[batch1], 0, len-len0);
      start = 0;
    } else {
      _xbatch = _x[batch0];  // taking this outside the loop does indeed make quite a big different (hotspot isn't catching this, then)
      _obatch = _o[batch0];
    }
    int offset = (int) (start % _batchSize);
    for (int i=1; i<len; i++) {
      int cmp = keycmp(_xbatch, offset+i, _xbatch, offset+i-1);  // TO DO: we don't need to compare the whole key here.  Set cmpLen < keySize
      if (cmp < 0) {
        System.arraycopy(_xbatch, (offset+i)*_keySize, keytmp, 0, _keySize);
        int j = i - 1;
        long otmp = _obatch[offset+i];
        do {
          System.arraycopy(_xbatch, (offset+j)*_keySize, _xbatch, (offset+j+1)*_keySize, _keySize);
          _obatch[offset+j+1] = _obatch[offset+j];
          j--;
        } while (j >= 0 && (cmp = keycmp(keytmp, 0, _xbatch, offset+j))<0);
        System.arraycopy(keytmp, 0, _xbatch, (offset+j+1)*_keySize, _keySize);
        _obatch[offset + j + 1] = otmp;
      }
      //if (cmp>0) _nGroup[_MSBvalue]++;  //TODO: reinstate _nGroup. Saves sweep afterwards. Possible now that we don't maintain the group sizes in this deep pass, unlike data.table
        // saves call to push() and hop to _groups
        // _nGroup == nrow afterwards tells us if the keys are unique.
        // Sadly, it seems _nGroup += (cmp==0) isn't possible in Java even with explicit cast of boolean to int, so branch needed
    }
    if (batch1 != batch0) {
      // Put the sorted data back into original two places straddling the boundary
      System.arraycopy(_x[batch0], (int)(origstart % _batchSize) *_keySize, _xbatch, 0, len0*_keySize);
      System.arraycopy(_x[batch1], 0, _xbatch, len0*_keySize, (len-len0)*_keySize);
      System.arraycopy(_o[batch0], (int)(origstart % _batchSize), _obatch, 0, len0);
      System.arraycopy(_o[batch1], 0, _obatch, len0, len-len0);
    }
  }

  public void run(long start, long len, int Byte) {

    // System.out.println("run " + start + " " + len + " " + Byte);
    if (len < 200) {               // N_SMALL=200 is guess based on limited testing. Needs calibrate().
      // Was 50 based on sum(1:50)=1275 worst -vs- 256 cummulate + 256 memset + allowance since reverse order is unlikely.
      insert(start, (int)len);   // when nalast==0, iinsert will be called only from within iradix.
      // TO DO:  inside insert it doesn't need to compare the bytes so far as they're known equal,  so pass Byte (NB: not Byte-1) through to insert()
      // TO DO:  Maybe transposing keys to be a set of _keySize byte columns might in fact be quicker - no harm trying. What about long and varying length string keys?
      return;
    }
    int batch0 = (int) (start / _batchSize);
    int batch1 = (int) (start+len-1) / _batchSize;
    // could well span more than one boundary when very large number of rows.
    // assert batch0==0;
    // assert batch0==batch1;  // Count across batches of 2Bn is now done.  Wish we had 64bit indexing in Java.
    long thisHist[] = counts[Byte];
    // thisHist reused and carefully set back to 0 below so we don't need to clear it now
    int idx = (int)(start%_batchSize)*_keySize + _keySize-Byte-1;
    int bin=-1;  // the last bin incremented. Just to see if there is only one bin with a count.
    int nbatch = batch1-batch0+1;  // number of batches this span of len covers.  Usually 1.  Minimum 1.
    int thisLen = (int)Math.min(len, _batchSize - start%_batchSize);
    for (int b=0; b<nbatch; b++) {
      byte _xbatch[] = _x[batch0+b];   // taking this outside the loop below does indeed make quite a big different (hotspot isn't catching this, then)
      for (int i = 0; i < thisLen; i++) {
        bin = 0xff & _xbatch[idx];
        thisHist[bin]++;
        idx += _keySize;
        // maybe TO DO:  shorten key by 1 byte on each iteration, so we only need to thisx && 0xFF.  No, because we need for construction of final table key columns.
      }
      idx = _keySize-Byte-1;
      thisLen = (b==nbatch-2/*next iteration will be last batch*/ ? (int)(start+len-1)%_batchSize : _batchSize);
      // thisLen will be set to _batchSize for the middle batches when nbatch>=3
    }
    if (thisHist[bin] == len) {
      // one bin has count len and the rest zero => next byte quick
      thisHist[bin] = 0;  // important, clear for reuse
      if (Byte == 0) ; // TODO: reinstate _nGroup[_MSBvalue]++;
      else run(start, len, Byte - 1);
      return;
    }
    long rollSum = 0;
    for (int c = 0; c < 256; c++) {
      long tmp = thisHist[c];
      if (tmp == 0) continue;  // important to skip zeros for logic below to undo cumulate. Worth the branch to save a deeply iterative memset back to zero
      thisHist[c] = rollSum;
      rollSum += tmp;
    }

    // Sigh. Now deal with batches here as well because Java doesn't have 64bit indexing.
    int oidx = (int)(start%_batchSize);
    int xidx = oidx*_keySize + _keySize-Byte-1;
    thisLen = (int)Math.min(len, _batchSize - start%_batchSize);
    for (int b=0; b<nbatch; b++) {
      long _obatch[] = _o[batch0+b];   // taking these outside the loop below does indeed make quite a big different (hotspot isn't catching this, then)
      byte _xbatch[] = _x[batch0+b];
      for (int i = 0; i < thisLen; i++) {
        long target = thisHist[0xff & _xbatch[xidx]]++;
        // now always write to the beginning of _otmp and _xtmp just to reuse the first hot pages
        _otmp[(int)(target/_batchSize)][(int)(target%_batchSize)] = _obatch[oidx+i];   // this must be kept in 8 bytes longs
        System.arraycopy(_xbatch, (oidx+i)*_keySize, _xtmp[(int)(target/_batchSize)], (int)(target%_batchSize)*_keySize, _keySize );
        xidx += _keySize;
        //  Maybe TO DO:  this can be variable byte width and smaller widths as descend through bytes (TO DO: reverse byte order so always doing &0xFF)
      }
      xidx = _keySize-Byte-1;
      oidx = 0;
      thisLen = (b==nbatch-2/*next iteration will be last batch*/ ? (int)(start+len-1)%_batchSize : _batchSize);
    }

    // now copy _otmp and _xtmp back over _o and _x from the start position, allowing for boundaries
    // _o, _x, _otmp and _xtmp all have the same _batchsize
    // Would be really nice if Java had 64bit indexing to save programmer time.
    long numRowsToCopy = len;
    int sourceBatch = 0, sourceOffset = 0;
    int targetBatch = (int)(start / _batchSize), targetOffset = (int)(start % _batchSize);
    int targetBatchRemaining = _batchSize - targetOffset;  // 'remaining' means of the the full batch, not of the numRowsToCopy
    int sourceBatchRemaining = _batchSize - sourceOffset;  // at most batchSize remaining.  No need to actually put the number of rows left in here
    int thisCopy;
    while (numRowsToCopy > 0) {   // TO DO: put this into class as well, to ArrayCopy into batched
      thisCopy = (int)Math.min(numRowsToCopy, Math.min(sourceBatchRemaining, targetBatchRemaining));
      System.arraycopy(_otmp[sourceBatch], sourceOffset,          _o[targetBatch], targetOffset,          thisCopy);
      System.arraycopy(_xtmp[sourceBatch], sourceOffset*_keySize, _x[targetBatch], targetOffset*_keySize, thisCopy*_keySize);
      numRowsToCopy -= thisCopy;
      // sourceBatch no change
      sourceOffset += thisCopy; sourceBatchRemaining -= thisCopy;
      targetOffset += thisCopy; targetBatchRemaining -= thisCopy;
      if (sourceBatchRemaining == 0) { sourceBatch++; sourceOffset = 0; sourceBatchRemaining = _batchSize; }
      if (targetBatchRemaining == 0) { targetBatch++; targetOffset = 0; targetBatchRemaining = _batchSize; }
      // 'source' and 'target' deliberately the same length variable names and long lines deliberately used so we
      // can easy match them up vertically to ensure they are the same
    }

    long itmp = 0;
    for (int i=0; i<256; i++) {
      if (thisHist[i]==0) continue;
      long thisgrpn = thisHist[i] - itmp;
      if (thisgrpn == 1 || Byte == 0) {
        //TODO reinstate _nGroup[_MSBvalue]++;
      } else {
        run(start+itmp, thisgrpn, Byte-1);
      }
      itmp = thisHist[i];
      thisHist[i] = 0;  // important, to save clearing counts on next iteration
    }
  }
}

public class RadixOrder {
  int _biggestBit[];
  int _bytesUsed[];
  //long[][][] _o;
  //byte[][][] _x;

  RadixOrder(Frame DF, boolean isLeft, int whichCols[]) {
    //System.out.println("Calling RadixCount ...");
    long t0 = System.nanoTime();
    _biggestBit = new int[whichCols.length];   // currently only biggestBit[0] is used
    _bytesUsed = new int[whichCols.length];
    //long colMin[] = new long[whichCols.length];
    for (int i = 0; i < whichCols.length; i++) {
      Vec col = DF.vec(whichCols[i]);
      //long range = (long) (col.max() - col.min());
      //assert range >= 1;   // otherwise log(0)==-Inf next line
      _biggestBit[i] = 1 + (int) Math.floor(Math.log(col.max()) / Math.log(2));   // number of bits starting from 1 easier to think about (for me)
      _bytesUsed[i] = (int) Math.ceil(_biggestBit[i] / 8.0);
      //colMin[i] = (long) col.min();   // TO DO: non-int/enum
    }
    if (_biggestBit[0] < 8) Log.warn("biggest bit should be >= 8 otherwise need to dip into next column (TODO)");  // TODO: feeed back to R warnings()
    int keySize = ArrayUtils.sum(_bytesUsed);   // The MSB is stored (seemingly wastefully on first glance) because we need it when aligning two keys in Merge()
    int batchSize = 256*1024*1024 / Math.max(keySize, 8) / 2 ;   // 256MB is the DKV limit.  / 2 because we fit o and x together in one OXBatch.
    // The Math.max ensures that batches of o and x are aligned, even for wide keys. To save % and / in deep iteration; e.g. in insert().
    System.out.println("Time to use rollup stats to determine biggestBit: " + (System.nanoTime() - t0) / 1e9);

    t0 = System.nanoTime();
    new RadixCount(isLeft, _biggestBit[0], whichCols[0]).doAll(DF.vec(whichCols[0]));
    System.out.println("Time of MSB count MRTask left local on each node (no reduce): " + (System.nanoTime() - t0) / 1e9);

    // NOT TO DO:  we do need the full allocation of x[] and o[].  We need o[] anyway.  x[] will be compressed and dense.
    // o is the full ordering vector of the right size
    // x is the byte key aligned with o
    // o AND x are what bmerge() needs. Pushing x to each node as well as o avoids inter-node comms.


    // System.out.println("Starting MSB hist reduce across nodes and SplitByMSBLocal MRTask ...");
    // Workaround for incorrectly blocking closeLocal() in MRTask is to do a double MRTask and pass a key between them to pass output
    // from first on that node to second on that node.  // TODO: fix closeLocal() blocking issue and revert to simpler usage of closeLocal()
    t0 = System.nanoTime();
    Key linkTwoMRTask = Key.make();
    SplitByMSBLocal tmp = new SplitByMSBLocal(isLeft, _biggestBit[0], keySize, batchSize, _bytesUsed, whichCols, linkTwoMRTask).doAll(DF.vecs(whichCols));   // postLocal needs DKV.put()
    System.out.println("SplitByMSBLocal MRTask (all local per node, no network) took : " + (System.nanoTime() - t0) / 1e9);
    System.out.print(tmp.profString());

    System.out.print("Starting SendSplitMSB ... ");
    t0 = System.nanoTime();
    new SendSplitMSB(linkTwoMRTask).doAllNodes();
    System.out.println("SendSplitMSB across all nodes took : " + (System.nanoTime() - t0) / 1e9);

    //long nGroup[] = new long[257];   // one extra for later to make undo of cumulate easier when finding groups.  TO DO: let grouper do that and simplify here to 256

    // dispatch in parallel
    RPC[] radixOrders = new RPC[256];
    System.out.print("Sending SingleThreadRadixOrder async RPC calls ... ");
    t0 = System.nanoTime();
    for (int i = 0; i < 256; i++) {
      //System.out.print(i+" ");
      radixOrders[i] = new RPC<>(SplitByMSBLocal.ownerOfMSB(i), new SingleThreadRadixOrder(DF, isLeft, batchSize, keySize, /*nGroup,*/ i)).call();
    }
    System.out.println("took : " + (System.nanoTime() - t0) / 1e9);

    System.out.print("Waiting for RPC SingleThreadRadixOrder to finish ... ");
    t0 = System.nanoTime();
    int i=0;
    for (RPC rpc : radixOrders) { //TODO: Use a queue to make this fully async
      // System.out.print(i+" ");
      rpc.get();
      //SingleThreadRadixOrder radixOrder = (SingleThreadRadixOrder)rpc.get();   // TODO: make sure all transient here
      i++;
    }
    System.out.println("took " + (System.nanoTime() - t0) / 1e9);

    // serial, do one at a time
//    for (int i = 0; i < 256; i++) {
//      H2ONode node = MoveByFirstByte.ownerOfMSB(i);
//      SingleThreadRadixOrder radixOrder = new RPC<>(node, new SingleThreadRadixOrder(DF, batchSize, keySize, nGroup, i)).call().get();
//      _o[i] = radixOrder._o;
//      _x[i] = radixOrder._x;
//    }

    // If sum(nGroup) == nrow then the index is unique.
    // 1) useful to know if an index is unique or not (when joining to it we know multiples can't be returned so can allocate more efficiently)
    // 2) If all groups are size 1 there's no need to actually allocate an all-1 group size vector (perhaps user was checking for uniqueness by counting group sizes)
    // 3) some nodes may have unique input and others may contain dups; e.g., in the case of looking for rare dups.  So only a few threads may have found dups.
    // 4) can sweep again in parallel and cache-efficient finding the groups, and allocate known size up front to hold the group sizes.
    // 5) can return to Flow early with the group count. User may now realise they selected wrong columns and cancel early.

  }
}
