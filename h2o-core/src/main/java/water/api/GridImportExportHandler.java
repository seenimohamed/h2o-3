package water.api;

import hex.Model;
import hex.grid.Grid;
import water.*;
import water.api.schemas3.GridExportV3;
import water.api.schemas3.GridImportV3;
import water.api.schemas3.KeyV3;
import water.fvec.persist.FramePersist;
import water.fvec.persist.PersistUtils;
import water.persist.Persist;
import water.util.FileUtils;

import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.util.Map;
import java.util.Objects;

import static hex.grid.Grid.REFERENCES_META_FILE_SUFFIX;

public class GridImportExportHandler extends Handler {

  /**
   * Loads a grid from a folder. Path to the folder and grid id (considered to be grid's filename) is required.
   * After a grid is loaded, an attempt to find all it's related models in the very same folder is made.
   * All models must be found in order to successfully import a grid. Grid's version must be the same as the version of
   * H2O it is imported into.
   *
   * @param version      API version
   * @param gridImportV3 Import arguments
   * @return Key to the imported Grid. Grid's key is the same as serialized in the binary file given.
   * @throws IOException Error reading grid or related models.
   */
  @SuppressWarnings("unused")
  public KeyV3.GridKeyV3 importGrid(final int version, final GridImportV3 gridImportV3) throws IOException {
    Objects.requireNonNull(gridImportV3);
    validateGridImportParameters(gridImportV3);

    final URI gridUri = FileUtils.getURI(gridImportV3.grid_path);
    if (!PersistUtils.exists(gridUri)) {
      throw new IllegalArgumentException("File not found " + gridUri);
    }
    final URI gridFramesUri = FileUtils.getURI(gridImportV3.grid_path + REFERENCES_META_FILE_SUFFIX);
    if (gridImportV3.load_frames && !PersistUtils.exists(gridFramesUri)) {
      throw new IllegalArgumentException("Requested to load with data frames, but the grid was saved without frames.");
    }
    final Persist persist = H2O.getPM().getPersistForURI(gridUri);
    try (final InputStream inputStream = persist.open(gridUri.toString())) {
      final AutoBuffer gridAutoBuffer = new AutoBuffer(inputStream);
      final Freezable freezable = gridAutoBuffer.get();
      if (!(freezable instanceof Grid)) {
        throw new IllegalArgumentException(String.format("Given file '%s' is not a Grid", gridImportV3.grid_path));
      }
      final Grid grid = (Grid) freezable;

      final String gridDirectory = persist.getParent(gridUri.toString());
      loadGridModels(grid, gridDirectory);
      if (gridImportV3.load_frames) {
        loadGridReferences(gridDirectory, gridFramesUri);
      }
      DKV.put(grid);
      return new KeyV3.GridKeyV3(grid._key);
    }

  }

  @SuppressWarnings("unused")
  public KeyV3.GridKeyV3 exportGrid(final int version, final GridExportV3 gridExportV3) throws IOException {
    validateGridExportParameters(gridExportV3);
    if(DKV.get(gridExportV3.grid_id) == null){
      throw new IllegalArgumentException(String.format("Grid with id '%s' has not been found.", gridExportV3.grid_id));
    }
    final Iced possibleGrid = DKV.get(gridExportV3.grid_id).get();
    if(!(possibleGrid instanceof Grid)){
      throw new IllegalArgumentException(String.format("Given Grid Key '%s' is not a valid Grid.", gridExportV3.grid_id));
    }

    final Grid serializedGrid = (Grid) possibleGrid;
    serializedGrid.exportBinary(gridExportV3.grid_directory);
    serializedGrid.exportModelsBinary(gridExportV3.grid_directory);
    if (gridExportV3.save_frames) {
      serializedGrid.exportReferencesBinary(gridExportV3.grid_directory);
    }

    return new KeyV3.GridKeyV3(serializedGrid._key);
  }


  /**
   * Basic sanity check for given Grid export parameters
   *
   * @param input An instance of {@link GridExportV3}, may not be null.
   */
  private void validateGridExportParameters(final GridExportV3 input) {
    Objects.requireNonNull(input);
    if (input.grid_directory == null || input.grid_directory.isEmpty()) {
      throw new IllegalArgumentException(String.format("Given grid directory '%s' is not a valid directory.",
              input.grid_directory));
    }

    if (input.grid_id == null || input.grid_id.isEmpty()) {
      throw new IllegalArgumentException(String.format("Invalid Grid id '%s'.", input.grid_id));
    }
  }

  /**
   * Basic sanity check for given Grid import parameters
   *
   * @param input An instance of {@link GridImportV3}, may not be null.
   */
  private void validateGridImportParameters(final GridImportV3 input) {
    Objects.requireNonNull(input);
    if (input.grid_path == null || input.grid_path.isEmpty()) {
      throw new IllegalArgumentException(String.format("Given grid directory '%s' is not a valid path.",
              input.grid_path));
    }
  }

  private static void loadGridModels(final Grid grid, final String gridDirectory) throws IOException {
    for (Key<Model> k : grid.getModelKeys()) {
      final Model<?, ?, ?> model = Model.importBinaryModel(gridDirectory + "/" + k.toString());
      assert model != null;
    }
  }
  
  private static void loadGridReferences(final String gridDirectory, final URI gridReferencesUri) {
    Map<String, String> referencesMap = PersistUtils.read(gridReferencesUri, AutoBuffer::get);
    final Futures fs = new Futures();
    referencesMap.forEach((key, type) -> {
      if ("frame".equals(type)) FramePersist.loadFrom(Key.make(key), gridDirectory).get();
      else PersistUtils.read(URI.create(gridDirectory + "/" + key), ab -> ab.getKey(Key.make(key), fs));
    });
    fs.blockForPending();
  }
}
