# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import logging
import os
import re
import tempfile
import threading
import time
import traceback
from typing import Dict, List

import qt
import vtk
import vtkSegmentationCore
import requests
import SampleData
import slicer
from MONAILabelReviewerLib.ImageData import ImageData
from MONAILabelReviewerLib.ImageDataController import ImageDataController, ImageDataStatistics
from MONAILabelReviewerLib.MONAILabelReviewerEnum import Label, Level, SegStatus
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin


class Reviewer:
    """Local reviewer selection policy for the Slicer UI."""

    NON_HUMAN_REVIEWERS = {""}
    DEFAULT_REVIEWER = "Neuro"


class MONAILabelReviewer(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "MONAILabel Reviewer"
        self.parent.categories = ["Active Learning"]
        self.parent.dependencies = []
        self.parent.contributors = ["Minh Duc, Do (rAIdiance)"]
        self.parent.helpText = """
This module provides the user to review on segmentations on X-Ray-dicom images.
See more information in <a href="...">module documentation</a>.
"""
        self.parent.acknowledgementText = """
Developed by rAiDiance, and  funded by Berlin Institute of Health (BIH).
"""


class MONAILabelReviewerWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation

        # Color set
        self.colorGreenPressedButton = "background-color : rgb(0, 250, 146)"
        self.colorDarkGrayButton = "background-color : rgb(169, 169, 169)"
        self.colorGreenButtonAfterSuccessfulLoad = "background-color : rgb(0, 144, 81)"
        self.colorGreenEasyButton = "background-color : rgb(0, 250, 146)"
        self.colorYellowMediumButton = "background-color : rgba(255, 251, 0, 179)"
        self.colorRedHardButton = "background-color : rgba(255, 38, 0, 179)"
        self.colorLightGreenButton = "background-color : rgb(115, 250, 121)"
        self.colorRedReviewerModeButton = "background-color : rgb(255, 126, 121)"
        self.colorBlueBasicModeButton = "background-color : rgb(118, 214, 255)"
        self.colorRed = "color: red"
        self.colorGreen = "color: green"
        self.colorLightYellow = "background-color : rgb(255,255,153)"

        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

        self.STATUS = SegStatus()
        self.LEVEL = Level()
        self.LABEL = Label()

        self.selectedReviewer: str = "Neuro"
        self.selectedClientId: str = ""
        self.currentImageId: str = ""
        self.currentLabelVersion: str = ""
        self.listImageData: List[ImageData] = None
        self.imageCounter: int = 0
        self.currentImageData: ImageData = None
        self.idToimageData: Dict[str, ImageData] = None

        # Meta Information
        self.finalStatus: str = ""
        self.finalLevel: str = ""
        self.finalComment: str = ""
        self.tmpdir = ""

        self.reviewersModeIsActive = False
        self.isSelectableByLabelVersion = False
        self._suppressUiSignals = False
        self._connectionInProgress = False
        self._connectionResult = None
        self._connectionTimer = None
        self._datastoreReady = False
        self._seriesLoaded = False
        self._connectionLogPath = os.path.join(tempfile.gettempdir(), "MONAILabelReviewer-connection.log")

        self.mapFiltersToBool: Dict[str, bool] = {
            "segmented": False,
            "notSegemented": False,
            "approved": False,
            "flagged": False,
        }

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/MONAILabelReviewer.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        # Bound the two expanding views inside Search Images. Without limits,
        # opening the collapsible section repeatedly recalculates the parent
        # layout and may consume all available memory in Slicer 5.12/Qt 5.
        self.ui.tabWidget.setFixedHeight(220)
        self.ui.textEdit_search.setFixedHeight(100)
        self.ui.tableWidge_imageMeta.setFixedHeight(220)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = MONAILabelReviewerLogic()

        # set segmentator editor
        self.segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
        self.addSegmentator()
        self.setLightVersion()

        self.ui.verticalLayout_10.addWidget(self.segmentEditorWidget)
        self.segmentEditorWidget.hide()
        self._lastSegmentationNode = None
        self._loadedReviewMeta = None
        self._segmentsTable = slicer.qMRMLSegmentsTableView()
        self._segmentsTable.setMRMLScene(slicer.mrmlScene)
        self._segmentsTable.setMinimumHeight(180)
        self.ui.verticalLayout_10.addWidget(self._segmentsTable)
        # 3D surfaces are an explicit, separate operation: building closed
        # surfaces runs on the GUI thread and may take seconds per series.
        self._build3DButton = qt.QPushButton("Показать 3D (построить поверхности)")
        self._build3DButton.setToolTip("Строит closed-surface только для текущей серии. Маски не меняются.")
        self._build3DButton.clicked.connect(self.onBuild3DSurfaces)
        self.ui.verticalLayout_10.addWidget(self._build3DButton)
        self.loadServerSelection()

        #  Section: Widget Elements
        self.ui.btn_connect_monai.clicked.connect(self.init_dicom_stream)
        self.ui.btn_load.clicked.connect(self.loadImageData)

        self.ui.btn_approved.clicked.connect(self.approveSegmentation)
        self.ui.btn_mark_revision.clicked.connect(self.flagSegmentation)

        self.ui.btn_next.clicked.connect(self.getNextSegmentation)
        self.ui.btn_previous.clicked.connect(self.getPreviousSegmenation)

        self.ui.btn_easy.clicked.connect(self.setEasy)
        self.ui.btn_medium.clicked.connect(self.setMedium)
        self.ui.btn_hard.clicked.connect(self.setHard)

        self.ui.btn_search.clicked.connect(self.search)
        self.ui.btn_search_annotator_reviewer.clicked.connect(self.searchByAnnotatorReviewer)
        self.ui.btn_search_level.clicked.connect(self.searchByLevel)

        self.ui.checkBox_search_approved.clicked.connect(self.checkedAppprovedSearch)
        self.ui.checkBox_search_flagged.clicked.connect(self.checkedFlaggedSearch)

        self.ui.btn_show_image.clicked.connect(self.showSearchedImage)

        self.ui.checkBox_flagged.clicked.connect(self.checkedFlagged)
        self.ui.checkBox_approved.clicked.connect(self.checkApproved)
        self.ui.checkBox_not_segmented.clicked.connect(self.checkNotSegmented)
        self.ui.checkBox_segmented.clicked.connect(self.checkSegmented)

        self.ui.btn_basic_mode.clicked.connect(self.setLightVersion)
        self.ui.btn_reviewers_mode.clicked.connect(self.setReviewerVersion)
        self.ui.comboBox_clients.currentIndexChanged.connect(self.index_changed)
        self.ui.comboBox_reviewers.currentIndexChanged.connect(self.indexReviewerchanged)

        self.ui.comboBox_label_version.currentIndexChanged.connect(self.indexLabelVersionChanged)
        self.ui.btn_save_new_version.clicked.connect(self.setSaveAsNewVersion)
        self.ui.btn_overwrite_version.clicked.connect(self.setOverwriteCurrentVersion)
        self.ui.btn_update_version.clicked.connect(self.updateAfterEditingSegmentation)
        self.ui.btn_edit_label.clicked.connect(self.displayEditorTools)
        self.ui.btn_delete_version.clicked.connect(self.setDeleteVersion)

    def getCurrentTime(self):
        return datetime.datetime.now()

    def getCurrentMetaStatus(self) -> str:
        if not self.finalStatus:
            return ""
        return self.finalStatus

    def setCurrentMetaStatus(self, status=""):
        self.finalStatus = status

    def getCurrentMetaLevel(self) -> str:
        if not self.finalLevel:
            return ""
        return self.finalLevel

    def setCurrentMetaLevel(self, level=""):
        self.finalLevel = level

    def getCurrentComment(self) -> str:
        comment = self.ui.plainText_comment.toPlainText()
        if not comment:
            return ""
        return comment

    def setCurrentComment(self, comment=""):
        self.finalComment = comment

    def getSelectedReviewer(self) -> str:
        selectedReviewer = self.ui.comboBox_reviewers.currentText
        if not selectedReviewer:
            return ""
        return selectedReviewer

    def getSelectedClientFromComboBox(self) -> str:
        selectedClient = self.ui.comboBox_clients.currentText
        if not selectedClient:
            return ""
        return selectedClient

    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()
        if self._connectionTimer is not None:
            self._connectionTimer.stop()

    def indexReviewerchanged(self, index):
        logging.info(f"{self.getCurrentTime()}: Selected reviewer: '{self.ui.comboBox_reviewers.currentText}'")
        self.selectedReviewer = self.ui.comboBox_reviewers.currentText

    def index_changed(self, index):
        if self._suppressUiSignals or self._connectionInProgress:
            logging.debug(f"{self.getCurrentTime()}: Ignoring client selection signal during initialization")
            return
        # Choosing an annotator only changes the pending filter. Downloading
        # the first MRI here froze Slicer for large Task04 data sets.
        self._seriesLoaded = False
        self.updateSectionAvailability()
        self.setLoadButtonColor(reload=True)

    def indexLabelVersionChanged(self, index):
        logging.debug(
            f"{self.getCurrentTime()}: Selected labal version: '{self.getCurrentLabelVersionFromComboBox()}', is enabled '{self.isSelectableByLabelVersion}'"
        )
        self.displayAdditionalMetaIfEdited(self.getCurrentLabelVersionFromComboBox())
        if self.isSelectableByLabelVersion:
            self.disableDifficultyButtons(tag=self.getCurrentLabelVersion())
            self.loadNextImage(imageData=self.currentImageData, tag=self.getCurrentLabelVersion())

    def getCurrentLabelVersionFromComboBox(self) -> str:
        labelString = self.ui.comboBox_label_version.currentText
        return self.parseSelectedVersionFromComboBox(labelString)

    def getCurrentLabelVersion(self) -> str:
        label = self.getCurrentLabelVersionFromComboBox()
        if label == "":
            label = self.LABEL.FINAL
        return label

    def disableDifficultyButtons(self, tag: str):
        if self.LABEL.VERSION in tag:
            self.ui.btn_easy.hide()
            self.ui.btn_medium.hide()
            self.ui.btn_hard.hide()
        else:
            self.ui.btn_easy.show()
            self.ui.btn_medium.show()
            self.ui.btn_hard.show()

    def setSaveAsNewVersion(self) -> bool:
        setToSave = bool(self.ui.btn_save_new_version.isChecked())
        self.ui.btn_update_version.enabled = True
        if setToSave:
            self.ui.btn_overwrite_version.setChecked(False)
            self.ui.btn_overwrite_version.setStyleSheet(self.colorDarkGrayButton)

            self.ui.btn_delete_version.setChecked(False)
            self.ui.btn_delete_version.setStyleSheet(self.colorDarkGrayButton)

            self.ui.btn_save_new_version.setStyleSheet(self.colorGreenPressedButton)
            self.ui.btn_update_version.setText("Confirm: Saving")
        return setToSave

    def setOverwriteCurrentVersion(self) -> bool:
        setToOverwrite = bool(self.ui.btn_overwrite_version.isChecked())
        self.ui.btn_update_version.enabled = True
        if setToOverwrite:
            self.ui.btn_save_new_version.setChecked(False)
            self.ui.btn_save_new_version.setStyleSheet(self.colorDarkGrayButton)

            self.ui.btn_delete_version.setChecked(False)
            self.ui.btn_delete_version.setStyleSheet(self.colorDarkGrayButton)

            self.ui.btn_overwrite_version.setStyleSheet(self.colorGreenPressedButton)
            self.ui.btn_update_version.setText("Confirm: Overwriting")

        return setToOverwrite

    def setDeleteVersion(self) -> bool:
        setToDelete = bool(self.ui.btn_delete_version.isChecked())
        self.ui.btn_update_version.enabled = True
        if setToDelete:
            self.ui.btn_save_new_version.setChecked(False)
            self.ui.btn_save_new_version.setStyleSheet(self.colorDarkGrayButton)

            self.ui.btn_overwrite_version.setChecked(False)
            self.ui.btn_overwrite_version.setStyleSheet(self.colorDarkGrayButton)

            self.ui.btn_delete_version.setStyleSheet(self.colorGreenPressedButton)
            self.ui.btn_update_version.setText("Confirm: Deletion")

        return setToDelete

    def setButtonColorReviewerOrBasicMode(self, isReviewerMode: bool):
        if isReviewerMode:
            self.ui.btn_reviewers_mode.setStyleSheet(self.colorRedReviewerModeButton)
            self.ui.btn_basic_mode.setStyleSheet(self.colorDarkGrayButton)
        else:
            self.ui.btn_reviewers_mode.setStyleSheet(self.colorDarkGrayButton)
            self.ui.btn_basic_mode.setStyleSheet(self.colorBlueBasicModeButton)

    def setReviewerVersion(self):
        self.reviewersModeIsActive = True
        self.setButtonColorReviewerOrBasicMode(isReviewerMode=self.reviewersModeIsActive)
        # section: Server
        # Reviewer Field
        self.ui.label_20.show()
        self.ui.comboBox_reviewers.show()

        # Approved bar
        self.ui.label_17.show()
        self.ui.progressBar_approved_total.show()
        self.ui.label_idx_appr_image.show()

        # section: Data set explorer

        # Approved bar
        self.ui.label_10.show()
        self.ui.progressBar_approved_client.show()
        self.ui.label_idx_appr_image_client.show()

        # filter option
        self.ui.label_6.show()
        self.ui.checkBox_not_segmented.show()
        self.ui.checkBox_flagged.show()
        self.ui.checkBox_segmented.show()
        self.ui.checkBox_approved.show()

        # section: Data evaluation
        self.ui.btn_easy.show()
        self.ui.btn_medium.show()
        self.ui.btn_hard.show()
        self.ui.label_level_difficulty.show()
        self.ui.btn_mark_revision.show()
        self.ui.btn_approved.show()

        # imag information
        self.ui.label_14.show()
        self.ui.lineEdit_status.show()
        self.ui.label_16.show()
        self.ui.lineEdit_level.show()
        self.ui.plainText_comment.show()
        if self.ui.btn_basic_mode.isChecked():
            self.ui.btn_basic_mode.setChecked(False)

        self.collapseAllSecions()

        self.activateSegmentatorEditor(activated=False)
        self.hideEditingSelectionOption(isHidden=False)

    # Section:  Light version Option
    def setLightVersion(self):
        self.reviewersModeIsActive = False
        self.setButtonColorReviewerOrBasicMode(isReviewerMode=self.reviewersModeIsActive)
        # section: Server
        # Reviewer Field
        self.ui.label_20.hide()
        self.ui.comboBox_reviewers.hide()

        # Approved bar
        self.ui.label_17.hide()
        self.ui.progressBar_approved_total.hide()
        self.ui.label_idx_appr_image.hide()

        # section: Data set explorer
        # Approved bar
        self.ui.label_10.hide()
        self.ui.progressBar_approved_client.hide()
        self.ui.label_idx_appr_image_client.hide()

        # filter option
        self.ui.label_6.hide()
        self.ui.checkBox_not_segmented.hide()
        self.ui.checkBox_flagged.hide()
        self.ui.checkBox_segmented.hide()
        self.ui.checkBox_approved.hide()

        # section: Data evaluation
        self.ui.btn_easy.hide()
        self.ui.btn_medium.hide()
        self.ui.btn_hard.hide()
        self.ui.label_level_difficulty.hide()
        self.ui.btn_mark_revision.hide()
        self.ui.btn_approved.hide()

        # imag information
        self.ui.label_14.hide()
        self.ui.lineEdit_status.hide()
        self.ui.label_16.hide()
        self.ui.lineEdit_level.hide()
        self.ui.plainText_comment.hide()
        if self.ui.btn_reviewers_mode.isChecked():
            self.ui.btn_reviewers_mode.setChecked(False)

        self.collapseAllSecions()

        self.activateSegmentatorEditor(activated=False)
        # Basic mode also supports local label correction and adding segments.
        # Reviewer-only filters and approval controls remain hidden.
        self.hideEditingSelectionOption(isHidden=False)

    def cleanCache(self):
        self.logic = MONAILabelReviewerLogic()
        self.selectedReviewer = Reviewer.DEFAULT_REVIEWER
        self.selectedClientId = ""
        self._datastoreReady = False
        self._seriesLoaded = False

        self.listImageData = None
        self.imageCounter = 0
        self.currentImageData = None
        self.idToimageData = None

        # Meta Information
        self.setCurrentMetaStatus(status="")
        self.setCurrentMetaLevel(level="")
        self.setCurrentComment(comment="")

        logging.info(f"{self.getCurrentTime()}: Cache is cleaned")

    # Section: Server
    def loadServerSelection(self):
        settings = qt.QSettings()
        serverUrlHistory = settings.value("MONAILabel/serverUrlHistory")

        self.ui.comboBox_server_url.clear()
        self.ui.comboBox_server_url.addItems(serverUrlHistory.split(";"))

    def init_dicom_stream(self):
        """
        initiates connection to monai server
        Default: client listens on "http://127.0.0.1:8000"
        """
        if self._connectionInProgress:
            return
        self._connectionInProgress = True
        try:
            self._init_dicom_stream_impl()
        except Exception:
            self._logConnection("Connection setup failed: " + traceback.format_exc())
            self._connectionInProgress = False
            slicer.util.warningDisplay("Reviewer connection setup failed. See MONAILabelReviewer-connection.log in the temporary folder.")

    def _logConnection(self, message):
        line = f"{datetime.datetime.now().isoformat(timespec='milliseconds')} {message}"
        logging.info("[Reviewer connection] %s", message)
        try:
            with open(self._connectionLogPath, "a", encoding="utf-8") as stream:
                stream.write(line + "\n")
        except OSError:
            logging.exception("Cannot write Reviewer connection log")

    def _init_dicom_stream_impl(self):
        # Only the worker uses requests and parses the datastore. Slicer and Qt
        # objects are accessed exclusively from the GUI thread in _finishConnection.
        self.cleanCache()
        serverUrl: str = self.ui.comboBox_server_url.currentText
        self._connectionResult = None
        self.ui.btn_connect_monai.enabled = False

        def worker():
            try:
                started = time.monotonic()
                controller = ImageDataController()
                connected = controller.connectToMonaiServer(serverUrl)
                if not connected:
                    self._connectionResult = (False, "Server did not respond", None)
                    return
                started = time.monotonic()
                loaded = controller.initMetaDataProcessing()
                if not loaded:
                    self._connectionResult = (False, "Datastore could not be read", None)
                    return
                started = time.monotonic()
                statistics = controller.getStatistics()
                clients = controller.getClientIds()
                reviewers = controller.getReviewers()
                self._connectionResult = (True, "", (controller, statistics, clients, reviewers))
            except Exception:
                error = traceback.format_exc()
                self._logConnection("WORKER ERROR " + error)
                self._connectionResult = (False, error, None)

        self._connectionTimer = qt.QTimer()
        self._connectionTimer.setInterval(100)
        self._connectionTimer.timeout.connect(self._finishConnection)
        self._connectionTimer.start()
        threading.Thread(target=worker, name="ReviewerDatastore", daemon=True).start()

    def _finishConnection(self):
        result = self._connectionResult
        if result is None:
            return
        self._connectionTimer.stop()
        self._connectionResult = None
        try:
            success, error, payload = result
            if not success:
                self._logConnection("FAILED " + error)
                slicer.util.warningDisplay("Reviewer connection failed. See MONAILabelReviewer-connection.log in the temporary folder.")
                return
            controller, statistics, clients, reviewers = payload
            self.logic.imageDataController = controller
            self._datastoreReady = True
            self._seriesLoaded = False
            self.ui.btn_connect_monai.setStyleSheet(self.colorGreenButtonAfterSuccessfulLoad)
            self.initUI(statistics=statistics, clients=clients, reviewers=reviewers)
        except Exception:
            self._logConnection("GUI ERROR " + traceback.format_exc())
            slicer.util.warningDisplay("Reviewer GUI update failed. See MONAILabelReviewer-connection.log in the temporary folder.")
        finally:
            self.ui.btn_connect_monai.enabled = True
            self._connectionInProgress = False

    def collapseAllSecions(self):
        self.ui.collapsibleButton_search_image.collapsed = True
        self.ui.collapsibleButton_dicom_stream.collapsed = True
        self.ui.collapsibleButton_dicom_evaluation.collapsed = True
        self.updateSectionAvailability()

    def updateSectionAvailability(self):
        """Keep mode switches from disabling an already connected session."""
        datastoreReady = bool(getattr(self, "_datastoreReady", False))
        seriesLoaded = bool(getattr(self, "_seriesLoaded", False))
        self.ui.collapsibleButton_search_image.enabled = datastoreReady
        self.ui.collapsibleButton_dicom_stream.enabled = datastoreReady
        self.ui.collapsibleButton_dicom_evaluation.enabled = datastoreReady and seriesLoaded

    def initUI(self, statistics=None, clients=None, reviewers=None):
        self._datastoreReady = True
        self.updateSectionAvailability()

        # set Segmentation progress bar
        self.setProgessBar(statistics=statistics)

        # Prevent combo-box signals from starting image loading while the UI
        # is still being populated.
        self._suppressUiSignals = True
        try:
            self.fillComboBoxes(clients=clients, reviewers=reviewers)
        finally:
            self._suppressUiSignals = False
        self.selectedReviewer = self.ui.comboBox_reviewers.currentText or Reviewer.DEFAULT_REVIEWER

        # set up buttons
        self.setButtons()

        self.selectedClientId = ""

    def setButtons(self):
        self.ui.btn_approved.setCheckable(True)
        self.ui.btn_mark_revision.setCheckable(True)
        self.ui.btn_easy.setCheckable(True)
        self.ui.btn_medium.setCheckable(True)
        self.ui.btn_hard.setCheckable(True)
        self.ui.btn_reviewers_mode.setCheckable(True)
        self.ui.btn_basic_mode.setCheckable(True)

        self.ui.btn_edit_label.setCheckable(True)
        self.ui.btn_save_new_version.setCheckable(True)
        self.ui.btn_overwrite_version.setCheckable(True)
        self.ui.btn_delete_version.setCheckable(True)
        self.ui.btn_update_version.setCheckable(True)

        self.ui.btn_delete_version.hide()
        self.ui.btn_save_new_version.hide()
        self.ui.btn_overwrite_version.hide()
        self.ui.btn_update_version.hide()
        self.ui.btn_update_version.enabled = False

        self.ui.btn_show_image.enabled = False

    def setProgessBar(self, statistics=None):
        if statistics is None:
            statistics = self.logic.getStatistics()

        self.ui.progressBar_segmentation.setProperty("value", statistics.getSegmentationProgress())
        self.ui.label_idx_seg_image.setText(statistics.getIdxTotalSegmented())
        self.ui.label_idx_appr_image.setText(statistics.getIdxTotalApproved())
        self.ui.progressBar_approved_total.setProperty("value", statistics.getProgressPercentage())

    def loadConfiguredReviewers(self):
        """Compatibility hook for versions that support local reviewer config."""
        return []

    def mergeReviewers(self, fromDatastore=None):
        """Put the local default first, then configured and datastore reviewers."""
        if fromDatastore is None:
            fromDatastore = self.logic.getReviewers()
        merged = [Reviewer.DEFAULT_REVIEWER] + self.loadConfiguredReviewers() + list(fromDatastore)
        return list(dict.fromkeys(
            reviewer for reviewer in merged
            if reviewer not in Reviewer.NON_HUMAN_REVIEWERS
        ))

    def fillComboBoxes(self, clients=None, reviewers=None):
        # clients
        clientIds = clients if clients is not None else self.logic.getClientIds()

        self.ui.comboBox_clients.clear()
        self.ui.comboBox_clients.addItem("All")
        for clientId in clientIds:
            self.ui.comboBox_clients.addItem(str(clientId))

        # combobox in search section
        self.ui.comboBox_search_annotator.clear()
        self.ui.comboBox_search_annotator.addItem("All")
        for clientId in clientIds:
            self.ui.comboBox_search_annotator.addItem(str(clientId))

        # reviewers
        self.ui.comboBox_reviewers.clear()
        reviewers = self.mergeReviewers(fromDatastore=reviewers)
        self.ui.comboBox_reviewers.addItems([str(reviewer) for reviewer in reviewers])
        defaultIndex = self.ui.comboBox_reviewers.findText(Reviewer.DEFAULT_REVIEWER)
        if defaultIndex >= 0:
            self.ui.comboBox_reviewers.setCurrentIndex(defaultIndex)
        self.selectedReviewer = Reviewer.DEFAULT_REVIEWER

        # combobox in search section
        self.ui.comboBox_search_reviewer.clear()
        self.ui.comboBox_search_reviewer.addItem("All")
        for reviewer in reviewers:
            self.ui.comboBox_search_reviewer.addItem(str(reviewer))

    def cleanDicomStreamSection(self):
        self.setCurrentMetaStatus(status="")
        self.setCurrentMetaLevel(level="")
        self.setCurrentComment(comment="")

        self.selectedClientId = None
        self.imageCounter = 0
        self.currentImageData = None
        self.idToimageData = None
        self.listImageData = None

        self.cleanProgressBarDicomStreamSection()
        self.cleanCheckBoxes()
        self.resetHorizontalSlider()

    # Section: Loading images
    def loadImageData(self):
        self.imageCounter = 0

        self.cleanSearchSection()
        # select segmentator: ALL
        self.selectedClientId = self.getSelectedClientFromComboBox()
        if self.selectedClientId == "All":
            self.listImageData = self.loadImageDataWithFilter(selectedClientId="")
            self.ui.checkBox_segmented.setEnabled(True)
            self.ui.checkBox_not_segmented.setEnabled(True)
            self.setProgressBarOfAll()

        # select segmentator: client was selected
        if self.selectedClientId != "All":
            self.listImageData = self.loadImageDataWithFilter(selectedClientId=self.selectedClientId)
            self.setCheckBoxesClient()
            self.setProgressBarOfClient(self.selectedClientId)

        logging.info(
            "{}: Successfully loaded Image data [total = {}, category = '{}']".format(
                self.getCurrentTime(), len(self.listImageData), self.selectedClientId
            )
        )

        loaded = False
        if len(self.listImageData) > 0:
            self.currentImageData = self.listImageData[self.imageCounter]
            loaded = self.loadNextImage(self.currentImageData)

        self.ui.collapsibleButton_dicom_evaluation.enabled = loaded
        if loaded:
            self.ui.collapsibleButton_dicom_evaluation.collapsed = False
        self.setHorizontalSlider(len(self.listImageData))
        self.collectFilters()
        self.setLoadButtonColor(reload=False)

    def loadImageDataWithFilter(self, selectedClientId: str) -> list:
        isApproved = bool(self.ui.checkBox_approved.isChecked())
        isFlagged = bool(self.ui.checkBox_flagged.isChecked())
        isNotSegmented = bool(self.ui.checkBox_not_segmented.isChecked())
        segmented = bool(self.ui.checkBox_segmented.isChecked())
        logging.info(
            "{}: Selected filters: segmented= {} | isNotSegmented= {} | isApproved= {} | isFlagged= {}".format(
                self.getCurrentTime(), segmented, isNotSegmented, isApproved, isFlagged
            )
        )
        if selectedClientId == "":
            return self.logic.getAllImageData(segmented, isNotSegmented, isApproved, isFlagged)
        return self.logic.getImageDataByClientId(selectedClientId, isApproved, isFlagged)

    def setProgressBarOfAll(self):
        statistics: ImageDataStatistics = self.logic.getStatistics()
        # Progress bar: Segmented/TotalImage
        self.ui.progressBar_segmented_client.setProperty("value", statistics.getSegmentationProgressAllPercentage())
        self.ui.label_idx_seg_image_client.setText(statistics.getIdxTotalSegmented())
        # Progress bar: approvalCount/TotalImage
        self.ui.progressBar_approved_client.setProperty("value", statistics.getApprovalProgressPercentage())
        self.ui.label_idx_appr_image_client.setText(statistics.getIdxTotalApproved())

    def cleanProgressBarDicomStreamSection(self):
        self.ui.progressBar_segmented_client.setProperty("value", 0)
        self.ui.progressBar_approved_client.setProperty("value", 0)
        self.ui.label_idx_seg_image_client.setText("x/y")
        self.ui.label_idx_appr_image_client.setText("x/y")

    def setLoadButtonColor(self, reload: bool):
        if reload:  # reload required
            self.ui.btn_load.setStyleSheet(self.colorDarkGrayButton)
            return
        self.ui.btn_load.setStyleSheet(self.colorGreenButtonAfterSuccessfulLoad)

    def setProgressBarOfClient(self, selectedClientId: str):
        percentageApprovedOfClient, idxApprovedOfClient = self.logic.getPercentageApproved(selectedClientId)
        self.ui.progressBar_approved_client.setProperty("value", percentageApprovedOfClient)
        self.ui.label_idx_appr_image_client.setText(idxApprovedOfClient)

        percentageSemgmentedByClient, idxSegmentedByClient = self.logic.getPercentageSemgmentedByClient(
            selectedClientId
        )
        self.ui.progressBar_segmented_client.setProperty("value", percentageSemgmentedByClient)
        self.ui.label_idx_seg_image_client.setText(idxSegmentedByClient)

    def setHorizontalSlider(self, loadesImageCount: int):
        self.ui.horizontalSlider_image_idx.setMinimum(0)
        self.ui.horizontalSlider_image_idx.setMaximum(max(0, loadesImageCount - 1))
        idxImage = f"Image: {self.imageCounter + 1 if loadesImageCount else 0}/{loadesImageCount}"
        self.ui.label_idx_image.setText(idxImage)

    def updateHorizontalSlider(self):
        self.ui.horizontalSlider_image_idx.setValue(self.imageCounter)
        idxImage = f"Image: {self.imageCounter + 1}/{len(self.listImageData)}"
        self.ui.label_idx_image.setText(idxImage)

    def resetHorizontalSlider(self):
        self.ui.horizontalSlider_image_idx.setValue(1)
        self.ui.label_idx_image.setText("Image:")

    # Section: Filter
    def collectFilters(self):
        self.mapFiltersToBool["segmented"] = self.ui.checkBox_segmented.isChecked()
        self.mapFiltersToBool["notSegemented"] = self.ui.checkBox_not_segmented.isChecked()
        self.mapFiltersToBool["approved"] = self.ui.checkBox_approved.isChecked()
        self.mapFiltersToBool["flagged"] = self.ui.checkBox_flagged.isChecked()

    def isDifferentFilter(self) -> bool:
        if self.mapFiltersToBool["segmented"] != self.ui.checkBox_segmented.isChecked():
            return True
        if self.mapFiltersToBool["notSegemented"] != self.ui.checkBox_not_segmented.isChecked():
            return True
        if self.mapFiltersToBool["approved"] != self.ui.checkBox_approved.isChecked():
            return True
        if self.mapFiltersToBool["flagged"] != self.ui.checkBox_flagged.isChecked():
            return True
        return False

    # CheckBox: clean
    def cleanCheckBoxes(self):
        self.ui.checkBox_segmented.setChecked(False)
        self.ui.checkBox_not_segmented.setChecked(False)
        self.ui.checkBox_flagged.setChecked(False)
        self.ui.checkBox_approved.setChecked(False)

    # CheckBox: flagged
    def setCheckBoxesClient(self):
        self.setLoadButtonColor(reload=True)
        self.ui.checkBox_not_segmented.setEnabled(False)
        self.ui.checkBox_segmented.setChecked(True)
        self.ui.checkBox_segmented.setEnabled(False)

    # CheckBox: flagged
    def checkedFlagged(self):
        self.setLoadButtonColor(reload=True)
        self.ui.checkBox_segmented.setChecked(True)
        if self.ui.checkBox_approved.isChecked():
            self.ui.checkBox_approved.setChecked(False)
        if self.ui.checkBox_not_segmented.isChecked():
            self.ui.checkBox_not_segmented.setChecked(False)

    # CheckBox: approved
    def checkApproved(self):
        self.setLoadButtonColor(reload=True)
        self.ui.checkBox_segmented.setChecked(True)
        if self.ui.checkBox_flagged.isChecked():
            self.ui.checkBox_flagged.setChecked(False)
        if self.ui.checkBox_not_segmented.isChecked():
            self.ui.checkBox_not_segmented.setChecked(False)

    # CheckBox: NOT segmented
    def checkNotSegmented(self):
        self.setLoadButtonColor(reload=True)
        if self.ui.checkBox_approved.isChecked():
            self.ui.checkBox_approved.setChecked(False)
        if self.ui.checkBox_flagged.isChecked():
            self.ui.checkBox_flagged.setChecked(False)
        if self.ui.checkBox_segmented.isChecked():
            self.ui.checkBox_segmented.setChecked(False)

    # CheckBox: segmented
    def checkSegmented(self):
        self.setLoadButtonColor(reload=True)
        if self.ui.checkBox_segmented.isChecked() is False:
            self.ui.checkBox_approved.setChecked(False)
            self.ui.checkBox_flagged.setChecked(False)
            return

        if self.ui.checkBox_not_segmented.isChecked():
            self.ui.checkBox_not_segmented.setChecked(False)

    # Section: Search Image
    def resetSearchSelection(self):
        self.currentImageData = None
        self.currentImageId = ""
        self.imageCounter = 0
        self.ui.collapsibleButton_dicom_evaluation.enabled = False
        self.ui.btn_show_image.enabled = False

    def cleanSearchSection(self):
        self.ui.tableWidge_imageMeta.setRowCount(0)
        self.ui.tableWidge_imageMeta.clearContents()
        self.ui.textEdit_search.clear()

    def search(self):
        """
        After triggering search button, load images and segmentation by input ids
        """
        self.cleanDicomStreamSection()
        self.resetSearchSelection()

        if self.ui.textEdit_search.toPlainText() == "":
            logging.info(f"{self.getCurrentTime()}: Search input field is empty")
            return

        idsStr = self.ui.textEdit_search.toPlainText()
        idList = self.getIdsFromString(idsStr)

        self.idToimageData = self.logic.getMultImageDataByIds(idList)
        self.listImageData = [*self.idToimageData.values()]

        foundIdList = [imageData.getName() for imageData in self.listImageData]
        notFoundIdList = [id for id in idList if (id not in foundIdList)]
        self.loadSearchImageMetaInTable(self.listImageData, notFoundIdList)

        self.setHorizontalSlider(len(foundIdList))
        if len(foundIdList) > 0:
            self.setSearchResultMessage(numOfFound=len(foundIdList))
            # Results are displayed first; Show loads the selected series.
        else:
            self.setSearchResultMessage(numOfFound=0)

    def searchByAnnotatorReviewer(self):
        self.resetSearchSelection()
        selectedAnnotator: str = self.ui.comboBox_search_annotator.currentText
        selectedReviewer: str = self.ui.comboBox_search_reviewer.currentText
        isApproved: bool = bool(self.ui.checkBox_search_approved.isChecked())
        isFlagged: bool = bool(self.ui.checkBox_search_flagged.isChecked())
        logging.warn(
            f"{self.getCurrentTime()}: Search by annontator: '{selectedAnnotator}' | reviewer: '{selectedReviewer}' | isApproved: '{isApproved}' | isFlagged: '{isFlagged}'"
        )

        self.idToimageData = self.logic.searchByAnnotatorReviewer(
            selectedAnnotator, selectedReviewer, isApproved, isFlagged
        )
        self.listImageData = [*self.idToimageData.values()]

        self.loadSearchImageMetaInTable(self.listImageData, [])
        if len(self.listImageData) > 0:
            self.setSearchResultMessage(numOfFound=len(self.idToimageData))
            self.setHorizontalSlider(len(self.idToimageData))
            # The user opens a chosen series with Show; searching must not
            # download an MRI or segmentation automatically.
        else:
            self.setSearchResultMessage(numOfFound=0)

    def searchByLevel(self):
        self.resetSearchSelection()
        isEasy: bool = bool(self.ui.checkBox_search_easy.isChecked())
        isMedium: bool = bool(self.ui.checkBox_search_medium.isChecked())
        isHard: bool = bool(self.ui.checkBox_search_hard.isChecked())

        self.idToimageData = self.logic.searchByLevel(isEasy, isMedium, isHard)
        self.listImageData = [*self.idToimageData.values()]

        self.loadSearchImageMetaInTable(self.listImageData, [])
        if len(self.listImageData) > 0:
            self.setSearchResultMessage(numOfFound=len(self.idToimageData))
            self.setHorizontalSlider(len(self.idToimageData))
            # Loading a series is an explicit action via Show.
        else:
            self.setSearchResultMessage(numOfFound=0)

    def setSearchResultMessage(self, numOfFound: int):
        if numOfFound == 0:
            self.ui.label_search_result.setText("Result: No images found.")
            self.ui.label_search_result.setStyleSheet(self.colorRed)
        else:
            resultMessage = f"Result: {numOfFound} images found."
            self.ui.label_search_result.setText(resultMessage)
            self.ui.label_search_result.setStyleSheet(self.colorGreen)

    def checkedAppprovedSearch(self):
        isFlagged = bool(self.ui.checkBox_search_flagged.isChecked())
        if isFlagged:
            self.ui.checkBox_search_flagged.setChecked(False)

    def checkedFlaggedSearch(self):
        isApproved = bool(self.ui.checkBox_search_approved.isChecked())
        if isApproved:
            self.ui.checkBox_search_approved.setChecked(False)

    def loadSearchImageMetaInTable(self, foundlist: List[ImageData], notFoundIdList: List[str]):
        """
        Set table content after triggering button "show"
        Parameters:
            foundlist (list): list contains found ids
            notFoundIdList (list): list contains not found ids
        """
        rowCount = len(foundlist) + len(notFoundIdList)
        self.ui.tableWidge_imageMeta.setSortingEnabled(False)
        self.ui.tableWidge_imageMeta.setRowCount(rowCount)
        rowCounter = 0
        for row, imageData in enumerate(foundlist):
            self.ui.tableWidge_imageMeta.setItem(row, 0, qt.QTableWidgetItem(imageData.getName()))
            self.ui.tableWidge_imageMeta.setItem(row, 1, qt.QTableWidgetItem("Yes"))
            self.ui.tableWidge_imageMeta.setItem(row, 2, qt.QTableWidgetItem(str(imageData.isSegemented())))
            rowCounter += 1

        for row, notFoundId in enumerate(notFoundIdList):
            self.ui.tableWidge_imageMeta.setItem(rowCounter, 0, qt.QTableWidgetItem(notFoundId))
            self.ui.tableWidge_imageMeta.setItem(rowCounter, 1, qt.QTableWidgetItem("No"))
            self.ui.tableWidge_imageMeta.setItem(rowCounter, 2, qt.QTableWidgetItem("No"))
            rowCounter += 1

        self.ui.btn_show_image.enabled = bool(foundlist)
        self.ui.tableWidge_imageMeta.setSortingEnabled(True)

    def loadFirstImage(self):
        self.imageCounter = 0
        self.currentImageData = self.listImageData[self.imageCounter]
        loaded = self.loadNextImage(self.currentImageData)
        self.ui.collapsibleButton_dicom_evaluation.enabled = loaded
        if loaded:
            self.setHorizontalSlider(len(self.listImageData))
            self.updateHorizontalSlider()

    def showSearchedImage(self):
        """
        displays dicom & segmentation to corresponding selected row in listed ids
        """
        selectedRow = self.ui.tableWidge_imageMeta.currentRow()
        if selectedRow == -1:
            logging.warn(f"{self.getCurrentTime()}: Selected row [row number = {selectedRow}]is not valid")
            return
        selectedImageId = self.ui.tableWidge_imageMeta.item(selectedRow, 0).text()

        if selectedImageId not in self.idToimageData:
            logging.info(f"{self.getCurrentTime()}: Selected image id [id = {selectedImageId}] was not found")
            return
        self.currentImageData = self.idToimageData[selectedImageId]
        self.imageCounter = self.listImageData.index(self.currentImageData)
        loaded = self.loadNextImage(self.currentImageData)
        self.ui.collapsibleButton_dicom_evaluation.enabled = loaded
        if loaded:
            self.setHorizontalSlider(len(self.listImageData))
            self.updateHorizontalSlider()

    def removeAllWhiteSpaces(self, strChain) -> str:
        """
        removes white spaces within string
        """
        pattern = r"\s+"
        return re.sub(pattern, "", strChain)

    def getIdsFromString(self, idStr: str) -> List[str]:
        """
        parses string which contains comma seperated ids
        Parameters:
            idStr (str): string which contains comma seperated ids
        Returns:
            list: contains ids
        """
        cleanedStr = self.removeAllWhiteSpaces(idStr)
        idsList = cleanedStr.split(",")
        return list(dict.fromkeys(idsList))  # remove all duplicates

    # Section: Dicom stream
    # Button: Approve
    def approveSegmentation(self):
        statusApproved = self.ui.btn_approved.isChecked()
        statusFlagged = self.ui.btn_mark_revision.isChecked()

        if statusFlagged or self.getCurrentMetaStatus() == self.STATUS.FLAGGED:
            self.ui.btn_mark_revision.setChecked(False)
            self.ui.btn_mark_revision.setDown(False)
            self.ui.btn_mark_revision.setStyleSheet(self.colorDarkGrayButton)

        if statusApproved:
            self.setCurrentMetaStatus(status=self.STATUS.APPROVED)
            self.ui.btn_approved.setChecked(True)
            self.ui.btn_approved.setStyleSheet(self.colorLightGreenButton)
            self.ui.btn_mark_revision.setStyleSheet(self.colorDarkGrayButton)
            self.ui.lineEdit_status.setStyleSheet(self.colorLightGreenButton)
        else:
            self.setCurrentMetaStatus(status="")
            self.resetButtonsOfApproveOrFlag()
        self.updateDisplayImageMetaData()

    # Button: Flagge
    def flagSegmentation(self):
        statusApproved = self.ui.btn_approved.isChecked()
        statusFlagged = self.ui.btn_mark_revision.isChecked()

        if statusApproved or self.getCurrentMetaStatus() == self.STATUS.APPROVED:
            self.ui.btn_approved.setChecked(False)
            self.ui.btn_approved.setDown(False)
            self.ui.btn_approved.setStyleSheet(self.colorDarkGrayButton)
        if statusFlagged:
            self.setCurrentMetaStatus(status=self.STATUS.FLAGGED)
            self.ui.btn_mark_revision.setChecked(True)
            self.ui.btn_mark_revision.setStyleSheet(self.colorLightGreenButton)
            self.ui.btn_approved.setStyleSheet(self.colorDarkGrayButton)
            self.ui.lineEdit_status.setStyleSheet(self.colorLightYellow)
        else:
            self.setCurrentMetaStatus(status="")
            self.resetButtonsOfApproveOrFlag()
        self.updateDisplayImageMetaData()

    def resetButtonsOfApproveOrFlag(self):
        self.ui.btn_mark_revision.setStyleSheet("")
        self.ui.btn_approved.setStyleSheet("")
        self.ui.lineEdit_status.setStyleSheet("")

    # Button: Clear
    def clearButtons(self):
        self.ui.btn_mark_revision.setChecked(False)
        self.ui.btn_approved.setChecked(False)

        self.ui.btn_mark_revision.setDown(False)
        self.ui.btn_approved.setDown(False)

        self.resetButtonsOfApproveOrFlag()

        self.ui.btn_easy.setChecked(False)
        self.ui.btn_medium.setChecked(False)
        self.ui.btn_hard.setChecked(False)

        self.ui.btn_easy.setDown(False)
        self.ui.btn_medium.setDown(False)
        self.ui.btn_hard.setDown(False)

        self.resetButtonOfDifficulty()

    def disableButtons(self):
        self.ui.btn_easy.setDown(False)
        self.ui.btn_medium.setDown(False)
        self.ui.btn_hard.setDown(False)

    def setDifficultyButtonAccordingColorAndChecked(self, difficulty: str):
        if difficulty == self.LEVEL.EASY:
            self.ui.btn_easy.setStyleSheet(self.colorGreenEasyButton)
            self.ui.lineEdit_level.setStyleSheet(self.colorGreenEasyButton)

            self.ui.btn_medium.setChecked(False)
            self.ui.btn_hard.setChecked(False)

            self.ui.btn_medium.setDown(False)
            self.ui.btn_hard.setDown(False)

            self.ui.btn_medium.setStyleSheet(self.colorDarkGrayButton)
            self.ui.btn_hard.setStyleSheet(self.colorDarkGrayButton)

        elif difficulty == self.LEVEL.MEDIUM:
            self.ui.btn_medium.setStyleSheet(self.colorYellowMediumButton)
            self.ui.lineEdit_level.setStyleSheet(self.colorYellowMediumButton)

            self.ui.btn_easy.setChecked(False)
            self.ui.btn_hard.setChecked(False)

            self.ui.btn_easy.setDown(False)
            self.ui.btn_hard.setDown(False)

            self.ui.btn_easy.setStyleSheet(self.colorDarkGrayButton)
            self.ui.btn_hard.setStyleSheet(self.colorDarkGrayButton)

        elif difficulty == self.LEVEL.HARD:
            self.ui.btn_hard.setStyleSheet(self.colorRedHardButton)
            self.ui.lineEdit_level.setStyleSheet(self.colorRedHardButton)

            self.ui.btn_easy.setChecked(False)
            self.ui.btn_medium.setChecked(False)

            self.ui.btn_easy.setDown(False)
            self.ui.btn_medium.setDown(False)

            self.ui.btn_easy.setStyleSheet(self.colorDarkGrayButton)
            self.ui.btn_medium.setStyleSheet(self.colorDarkGrayButton)

    # Button: Easy
    def setEasy(self):
        levelEasy = self.ui.btn_easy.isChecked()
        if levelEasy:
            self.setCurrentMetaLevel(level=self.LEVEL.EASY)
            self.setDifficultyButtonAccordingColorAndChecked(difficulty=self.LEVEL.EASY)
            self.ui.lineEdit_level.setText(self.getCurrentMetaLevel())

        if levelEasy is False and self.getCurrentMetaLevel() == self.LEVEL.EASY:
            self.setCurrentMetaLevel(level="")
            self.resetButtonOfDifficulty()
            self.ui.lineEdit_level.setStyleSheet("")

        self.updateDisplayImageMetaData()

    # Button: Medium
    def setMedium(self):
        levelMedium = self.ui.btn_medium.isChecked()
        if levelMedium:
            self.setCurrentMetaLevel(level=self.LEVEL.MEDIUM)
            self.setDifficultyButtonAccordingColorAndChecked(difficulty=self.LEVEL.MEDIUM)
            self.ui.lineEdit_level.setText(self.getCurrentMetaLevel())

        if levelMedium is False and self.getCurrentMetaLevel() == self.LEVEL.MEDIUM:
            self.setCurrentMetaLevel(level="")
            self.resetButtonOfDifficulty()
            self.ui.lineEdit_level.setStyleSheet("")

        self.updateDisplayImageMetaData()

    # Button: Hard
    def setHard(self):
        levelHard = self.ui.btn_hard.isChecked()

        if levelHard:
            self.setCurrentMetaLevel(level=self.LEVEL.HARD)
            self.setDifficultyButtonAccordingColorAndChecked(difficulty=self.LEVEL.HARD)
            self.ui.lineEdit_level.setText(self.getCurrentMetaLevel())

        if levelHard is False and self.getCurrentMetaLevel() == self.LEVEL.HARD:
            self.setCurrentMetaLevel(level="")
            self.resetButtonOfDifficulty()
            self.ui.lineEdit_level.setStyleSheet("")

        self.updateDisplayImageMetaData()

    def resetButtonOfDifficulty(self):
        self.ui.btn_easy.setStyleSheet(self.colorGreenEasyButton)
        self.ui.btn_medium.setStyleSheet(self.colorYellowMediumButton)
        self.ui.btn_hard.setStyleSheet(self.colorRedHardButton)

    # Button: Next
    def getNextSegmentation(self):
        """
        after triggering next button:
          1. persist meta data in monai server
          2. update progess bar
          3. load next dicom & segmentation
        """

        self.persistMetaInMonaiServer()

        # Re process Meta Data after image data was persisted
        self.reloadOverallStatistic()
        # Request Next Image
        self.imageCounter += 1

        if self.imageCounter >= len(self.listImageData):
            message = f"{self.getCurrentTime()}: End of list has been reached."
            slicer.util.warningDisplay(message)
            self.imageCounter = len(self.listImageData) - 1
            return
        self.updateHorizontalSlider()
        self.currentImageData = self.listImageData[self.imageCounter]

        # Displays Next Image
        self.loadNextImage(self.currentImageData)
        self.resetSegmentationEditorTools()
        self.activateSegmentatorEditor(activated=False)

    # Monai Server: Put
    def persistMetaInMonaiServer(self):
        """
        Sends the updated meta data of dicom and segmentation to monai-server
        Monai-server incorporates that information into datastore.json file

        Only explicit reviewer changes (approve/flag, level, comment) in
        Reviewer mode are sent.  Navigating with Next must not write
        anything to the server datastore.
        """
        if self.currentImageData is None:
            return
        if not self.reviewersModeIsActive:
            logging.info("Review meta not sent: Basic mode is read-only for the datastore")
            return
        currentMeta = (self.getCurrentMetaStatus(), self.getCurrentMetaLevel(), self.getCurrentComment())
        if currentMeta == self._loadedReviewMeta:
            logging.info("Review meta not sent: no reviewer changes for '%s'", self.currentImageData.getName())
            return
        self.logic.updateLabelInfo(
            imageData=self.currentImageData,
            versionTag=self.getCurrentLabelVersion(),
            status=self.getCurrentMetaStatus(),
            level=self.getCurrentMetaLevel(),
            approvedBy=self.selectedReviewer,
            comment=self.getCurrentComment(),
        )

    # Button: Previouse
    def getPreviousSegmenation(self):
        """
        Loads the previous dicom and corresponding segmentation
        after useres tiggers Previous-Button
        """
        self.imageCounter -= 1
        if self.imageCounter < 0:
            message = f"{self.getCurrentTime()}: Lower limit of data set has been reached."
            slicer.util.warningDisplay(message)
            self.imageCounter = 0
            return
        self.updateHorizontalSlider()
        self.currentImageData = self.listImageData[self.imageCounter]
        self.currentImageData.display()

        self.fillComboBoxLabelVersions(self.currentImageData)
        approvedOrLatestVersionTag = self.currentImageData.getApprovedVersionTagElseReturnLatestVersion()
        self.loadNextImage(self.currentImageData, tag=approvedOrLatestVersionTag)
        self.resetSegmentationEditorTools()

    def cleanLineEditsContainingSegMeta(self):
        self.ui.lineEdit_image_id.setText("")
        self.ui.lineEdit_status.setText("")
        self.ui.lineEdit_segmentator.setText("")
        self.ui.lineEdit_level.setText("")
        self.ui.lineEdit_level.setStyleSheet("")
        self.ui.lineEdit_date.setText("")
        self.ui.plainText_comment.setPlainText("")

    def displayImageMetaData(self, imageData: ImageData, currentLabelVersion: str):
        """
        Displays meta info of dicom and segmentation in the info box on slicer

        Parameters:
          imageData (ImageData): Contains meta data (of dicom and segmenation)
        """
        self.cleanLineEditsContainingSegMeta()
        self.clearButtons()

        self.setCurrentMetaStatus(status=imageData.getStatus(currentLabelVersion))

        self.fillLineEditsWithSegmenationMeta(imageData, currentLabelVersion)
        self.setMetaButtonsAccordingToImageData(imageData, currentLabelVersion)

        self.setCurrentMetaLevel(level=imageData.getLevel(currentLabelVersion))
        # Snapshot of the review meta as loaded from the server; Next only
        # persists when the reviewer changed something explicitly.
        self._loadedReviewMeta = (self.getCurrentMetaStatus(), self.getCurrentMetaLevel(), self.getCurrentComment())

    def setMetaButtonsAccordingToImageData(self, imageData: ImageData, currentLabelVersion: str):
        finalLevel = imageData.getLevel(currentLabelVersion)
        if finalLevel != "":
            self.activateBtnLevelOfDifficulty(finalLevel)

        if imageData.isApprovedVersion(currentLabelVersion):
            self.activateBtnApproved(True)

        if imageData.isFlagged(currentLabelVersion):
            self.activateBtnApproved(False)

    def fillLineEditsWithSegmenationMeta(self, imageData: ImageData, currentLabelVersion: str):
        logging.debug(f"==== currentLabelVersion: {currentLabelVersion}")
        logging.debug(f"==== getName: {imageData.getName()}")
        logging.debug(f"==== getClientId: {imageData.getClientId(currentLabelVersion)}")
        logging.debug(f"==== getTime: {imageData.getTimeOfAnnotation()}")
        logging.debug(f"==== getStatus: {imageData.getStatus(currentLabelVersion)}")
        logging.debug(f"==== getComment: {imageData.getComment(currentLabelVersion)}")
        logging.debug(f"==== getLevel: {imageData.getLevel(currentLabelVersion)}")
        logging.debug(f"==== edtitingTme: {imageData.getTimeOfEditing(currentLabelVersion)}")

        name = imageData.getName()
        annotator = imageData.getClientId(currentLabelVersion)
        editor = imageData.getApprovedBy(currentLabelVersion)
        edtitingTme = imageData.getTimeOfEditing(currentLabelVersion)
        annotationTime = imageData.getTimeOfAnnotation()
        status = imageData.getStatus(currentLabelVersion)
        comment = imageData.getComment(currentLabelVersion)

        self.ui.lineEdit_image_id.setText(name)
        self.ui.lineEdit_segmentator.setText(annotator)
        self.ui.lineEdit_editor.setText(editor)
        self.ui.lineEdit_editing_date.setText(edtitingTme)
        self.ui.lineEdit_date.setText(annotationTime)
        self.ui.lineEdit_status.setText(status)
        self.ui.plainText_comment.setPlainText(comment)

        finalLevel = imageData.getLevel(currentLabelVersion)
        self.ui.lineEdit_level.setText(finalLevel)

    def activateBtnLevelOfDifficulty(self, finalLevel):
        if finalLevel == self.LEVEL.EASY:
            self.ui.btn_easy.setDown(True)
            self.ui.btn_easy.setChecked(True)

            self.ui.btn_easy.setStyleSheet(self.colorGreenEasyButton)
            self.ui.btn_medium.setStyleSheet(self.colorDarkGrayButton)
            self.ui.btn_hard.setStyleSheet(self.colorDarkGrayButton)

            self.ui.lineEdit_level.setStyleSheet(self.colorGreenEasyButton)
            self.setEasy()

        elif finalLevel == self.LEVEL.MEDIUM:
            self.ui.btn_medium.setDown(True)
            self.ui.btn_medium.setChecked(True)

            self.ui.btn_easy.setStyleSheet(self.colorDarkGrayButton)
            self.ui.btn_medium.setStyleSheet(self.colorYellowMediumButton)
            self.ui.btn_hard.setStyleSheet(self.colorDarkGrayButton)

            self.ui.lineEdit_level.setStyleSheet(self.colorYellowMediumButton)
            self.setMedium()

        elif finalLevel == self.LEVEL.HARD:
            self.ui.btn_hard.setDown(True)
            self.ui.btn_hard.setChecked(True)

            self.ui.btn_easy.setStyleSheet(self.colorDarkGrayButton)
            self.ui.btn_medium.setStyleSheet(self.colorDarkGrayButton)
            self.ui.btn_hard.setStyleSheet(self.colorRedHardButton)

            self.ui.lineEdit_level.setStyleSheet(self.colorRedHardButton)
            self.setHard()

    def activateBtnApproved(self, activated: bool):
        self.ui.btn_mark_revision.setChecked(not activated)
        self.ui.btn_mark_revision.setDown(not activated)
        self.ui.btn_approved.setChecked(activated)
        self.ui.btn_approved.setDown(activated)
        if activated:
            self.ui.btn_approved.setStyleSheet(self.colorLightGreenButton)
            self.ui.btn_mark_revision.setStyleSheet(self.colorDarkGrayButton)
            self.ui.lineEdit_status.setStyleSheet(self.colorLightGreenButton)
        else:
            self.ui.btn_approved.setStyleSheet(self.colorDarkGrayButton)
            self.ui.btn_mark_revision.setStyleSheet(self.colorLightGreenButton)
            self.ui.lineEdit_status.setStyleSheet(self.colorLightYellow)

    def updateDisplayImageMetaData(self):
        """
        Displays updated level (easy, medium, hard)
        in the info box on slicer
        """
        self.ui.lineEdit_status.setText(self.getCurrentMetaStatus())

    def loadNextImage(self, imageData: ImageData, tag=""):
        """
        Loads original Dicom image and Segmentation into slicer window
        Parameters:
          imageData (ImageData): Contains meta data (of dicom and segmenation)
                                which is required for rest request to monai server
                                in order to get dicom and segmenation (.nrrd).
        """
        self.currentImageData = imageData
        self._seriesLoaded = False
        self.updateSectionAvailability()
        slicer.mrmlScene.Clear()
        self.clearInformationFields()

        if tag == "":
            tag = imageData.getApprovedVersionTagElseReturnLatestVersion()
        if tag == "":
            tag = self.getCurrentLabelVersion()
        logging.debug(f"{self.getCurrentTime()} Loading image (id='{imageData.getName()}', tag='{tag}')")
        self.disableDifficultyButtons(tag=tag)
        self.displayImageMetaData(imageData, tag)

        try:
            self.logic.loadDicomAndSegmentation(imageData, tag)
            _schemaNode = self.logic.ensureProjectSegmentSchema()
            # Reviewer's mode can refresh the Segment Editor after loading;
            # re-assert visibility so the label remains visible in 2D too.
            self.logic.ensureAllSegmentationViewsVisible()
            try:
                _segNode = _schemaNode or getattr(self.logic, "_lastSegmentationNode", None)
                if _segNode is not None and _segNode.GetScene() is None:
                    _segNode = None
                self._lastSegmentationNode = _segNode
                self._segmentsTable.setSegmentationNode(None)
                self._segmentsTable.setSegmentationNode(_segNode)
            except Exception:
                logging.exception("Could not show segment list")
        except Exception as exception:
            logging.exception(
                "%s: Failed to load selected series '%s'", self.getCurrentTime(), imageData.getName()
            )
            slicer.util.warningDisplay(
                "Не удалось загрузить выбранную серию или маску. "
                "Серия оставлена в списке; проверьте Python Console.\n\n"
                f"{exception}"
            )
            self._seriesLoaded = False
            self.updateSectionAvailability()
            return False

        serverUrl = self.logic.getServerUrl().rstrip("/").lower()
        isShoulderProject = (
            serverUrl.endswith(":8103")
            or serverUrl.endswith(":8104")
            or "/monai/task04" in serverUrl
            or "/monai/task05" in serverUrl
        )
        if imageData.getStatus() != self.STATUS.NOT_SEGMENTED or isShoulderProject:
            # Do not bind the embedded Segment Editor while a series is being
            # loaded.  In Slicer 5.12, assigning a large segmentation to the
            # editor during this path can block the GUI thread indefinitely.
            # The MRI and segmentation are already displayed above; the
            # editor is attached when the user explicitly starts editing.
            pass

        if self.currentImageId != imageData.getName():
            self.currentImageId = imageData.getName()
            self.fillComboBoxLabelVersions(imageData)
        self._seriesLoaded = True
        self.updateSectionAvailability()
        return True

    def fillComboBoxLabelVersions(self, imageData: ImageData):
        self.isSelectableByLabelVersion = False
        self.ui.comboBox_label_version.clear()
        labelVersions = imageData.getVersionNames()
        approvedVersion = ""
        labelVersion = ""
        for labelVersion in labelVersions:
            if imageData.isApprovedVersion(versionTag=labelVersion) is True:
                labelVersion = "{} ({})".format(labelVersion, "approved")
                approvedVersion = labelVersion
            self.ui.comboBox_label_version.addItem(labelVersion)
        if approvedVersion != "":
            self.setVersionTagInComboBox(approvedVersion)
        elif labelVersion != "":
            self.setVersionTagInComboBox(labelVersion)
        self.isSelectableByLabelVersion = True

    def setVersionTagInComboBox(self, versionTag="", currentImageData=None):
        if self.isBlank(versionTag):
            return
        if (currentImageData is not None) and (currentImageData.isApprovedVersion(versionTag=versionTag) is True):
            versionTag = "{} ({})".format(versionTag, "approved")
        self.ui.comboBox_label_version.setCurrentText(versionTag)

    def parseSelectedVersionFromComboBox(self, versionTagString: str) -> str:
        if self.isBlank(versionTagString):
            return ""
        array = versionTagString.split()
        if len(array) == 0:
            return ""
        return array[0]

    # Sub Section: Display version selection option
    def activateSegmentatorEditor(self, activated=False):
        self.segmentEditorWidget.setSourceVolumeNodeSelectorVisible(activated)
        self.segmentEditorWidget.setSegmentationNodeSelectorVisible(activated)
        self.segmentEditorWidget.setSwitchToSegmentationsButtonVisible(activated)
        self.segmentEditorWidget.unorderedEffectsVisible = activated
        self.segmentEditorWidget.setReadOnly(not activated)

    def displayEditorTools(self):
        isCheckedForEdit = self.ui.btn_edit_label.isChecked()
        if isCheckedForEdit:
            if not getattr(self, "_seriesLoaded", False) or self.currentImageData is None:
                self.ui.btn_edit_label.setChecked(False)
                slicer.util.warningDisplay("Сначала загрузите конкретную серию и её маску.")
                return
            try:
                # The embedded qMRMLSegmentEditorWidget is unsafe for this
                # project's multi-segment labels in Slicer 5.12: attaching
                # the segmentation can block the GUI thread.  Use Slicer's
                # maintained Segment Editor module instead.  It attaches the
                # same MRML segmentation and source volume through its own
                # supported lifecycle.  Only MRML references are set here
                # (no Qt widget binding), then the module is opened.
                slicer.util.selectModule("SegmentEditor")
                # The standard module owns its own qMRMLSegmentEditorWidget.
                # Attach the current MRI and segmentation after the module
                # has been activated, rather than only configuring Reviewer's
                # private parameter node.
                qt.QTimer.singleShot(0, self._attachStandardSegmentEditor)
            except Exception as exception:
                logging.exception("Could not open standard Segment Editor")
                self.ui.btn_edit_label.setChecked(False)
                slicer.util.warningDisplay(f"Не удалось открыть стандартный редактор сегментации:\n{exception}")
                return
            self._showBackToReviewerButton(True)
            self.clearButtons()
            self.ui.btn_edit_label.setText("Reset current label edit")
            self.cleanLineEditsContainingSegMetaWhenStartEditing()
            self.activatedEditorTools()
        else:
            self.deactivatedEditorTools()
            self.loadNextImage(imageData=self.currentImageData, tag=self.getCurrentLabelVersion())

    def _attachStandardSegmentEditor(self):
        """Attach the current series to Slicer's real Segment Editor widget."""
        segmentationNode = self._currentSegmentationNode()
        sourceVolumeNode = getattr(self.logic, "_lastSourceVolumeNode", None)
        if segmentationNode is None or segmentationNode.GetScene() is None:
            raise RuntimeError("Current segmentation node is not available")
        if sourceVolumeNode is not None and sourceVolumeNode.GetScene() is None:
            sourceVolumeNode = None

        representation = slicer.modules.segmenteditor.widgetRepresentation()
        editor = representation.self().editor
        editor.setMRMLScene(slicer.mrmlScene)
        editor.setSegmentationNode(segmentationNode)
        if sourceVolumeNode is not None:
            editor.setSourceVolumeNode(sourceVolumeNode)
        editor.setReadOnly(False)
        editor.updateWidgetFromMRML()
        logging.info(
            "EDIT: standard Segment Editor attached segmentation=%s source=%s",
            segmentationNode.GetName(),
            sourceVolumeNode.GetName() if sourceVolumeNode is not None else None,
        )

    def cleanLineEditsContainingSegMetaWhenStartEditing(self):
        self.setCurrentMetaStatus(status="")
        self.setCurrentComment(comment="")
        self.ui.lineEdit_status.setText("")
        self.ui.lineEdit_editing_date.setText("")
        self.ui.plainText_comment.setPlainText("")

    def clearInformationFields(self):
        self.setCurrentMetaStatus(status="")
        self.setCurrentComment(comment="")
        self.setCurrentMetaLevel(level="")

    def activatedEditorTools(self):
        self.activateSegmentatorEditor(activated=True)
        self.ui.btn_save_new_version.setChecked(False)
        self.ui.btn_overwrite_version.setChecked(False)
        self.ui.btn_delete_version.setChecked(False)

        self.ui.btn_save_new_version.show()
        self.ui.btn_save_new_version.setStyleSheet("")

        self.ui.btn_overwrite_version.show()
        self.ui.btn_overwrite_version.setStyleSheet("")

        self.ui.btn_delete_version.show()
        self.ui.btn_delete_version.setStyleSheet("")
        self.ui.btn_update_version.show()

        self.ui.btn_next.enabled = False
        self.ui.btn_previous.enabled = False
        self.ui.btn_easy.enabled = False
        self.ui.btn_medium.enabled = False
        self.ui.btn_hard.enabled = False
        self.ui.btn_mark_revision.enabled = False
        self.ui.btn_approved.enabled = False

    def _showBackToReviewerButton(self, visible: bool):
        """Button at the top of Segment Editor that returns to the Reviewer."""
        try:
            button = getattr(self, "_backToReviewerButton", None)
            if button is None:
                if not visible:
                    return
                editorPanel = slicer.modules.segmenteditor.widgetRepresentation()
                button = qt.QPushButton("\u2b05 Вернуться в MONAI Reviewer (сохранить правку)")
                button.setStyleSheet("background-color: rgb(118, 214, 255); font-weight: bold; padding: 6px;")
                button.clicked.connect(lambda: slicer.util.selectModule("MONAILabelReviewer"))
                editorPanel.layout().insertWidget(0, button)
                self._backToReviewerButton = button
            button.setVisible(visible)
        except Exception:
            logging.exception("Could not add back-to-Reviewer button")

    def _currentSegmentationNode(self):
        node = getattr(self, "_lastSegmentationNode", None)
        if node is not None and node.GetScene() is not None:
            return node
        return slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode")

    def _prepareStandardSegmentEditor(self):
        """Point the standard Segment Editor at the current series' nodes."""
        segmentationNode = self._currentSegmentationNode()
        sourceVolumeNode = getattr(self.logic, "_lastSourceVolumeNode", None)
        if sourceVolumeNode is not None and sourceVolumeNode.GetScene() is None:
            sourceVolumeNode = None
        parameterNode = self.selectParameterNode(bindToWidget=False)
        if segmentationNode is not None:
            parameterNode.SetAndObserveSegmentationNode(segmentationNode)
        if sourceVolumeNode is not None:
            parameterNode.SetAndObserveSourceVolumeNode(sourceVolumeNode)

    def onBuild3DSurfaces(self):
        """Explicit, separate 3D operation for the current series."""
        node = self._currentSegmentationNode()
        if node is None or not getattr(self, "_seriesLoaded", False):
            slicer.util.warningDisplay("Сначала загрузите серию с маской.")
            return
        self._build3DButton.enabled = False
        try:
            self.logic.buildClosedSurface(node)
        finally:
            self._build3DButton.enabled = True

    def _closeEmbeddedEditor(self):
        table = getattr(self, "_segmentsTable", None)
        if table is not None:
            table.show()

    def deactivatedEditorTools(self):
        self._showBackToReviewerButton(False)
        self._closeEmbeddedEditor()
        self.activateSegmentatorEditor(activated=False)
        self.ui.btn_save_new_version.setChecked(False)
        self.ui.btn_overwrite_version.setChecked(False)
        self.ui.btn_delete_version.setChecked(False)

        self.ui.btn_save_new_version.hide()
        self.ui.btn_overwrite_version.hide()
        self.ui.btn_delete_version.hide()
        self.ui.btn_update_version.hide()
        self.ui.btn_update_version.enabled = False

        self.ui.btn_next.enabled = True
        self.ui.btn_previous.enabled = True
        self.ui.btn_easy.enabled = True
        self.ui.btn_medium.enabled = True
        self.ui.btn_hard.enabled = True
        self.ui.btn_mark_revision.enabled = True
        self.ui.btn_approved.enabled = True

        self.ui.btn_update_version.setText("Confirm")
        self.ui.btn_edit_label.setText("Start label edit")

    def resetSegmentationEditorTools(self):
        self.ui.btn_edit_label.setChecked(False)
        self.deactivatedEditorTools()

    def hideEditingSelectionOption(self, isHidden: bool):
        if isHidden:
            self.ui.btn_edit_label.hide()
        else:
            self.ui.btn_edit_label.show()

    # Section: Display label
    def addSegmentator(self):
        self.segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
        # Keep the widget unbound until the user explicitly starts editing.
        # Binding an empty parameter node here makes later scene replacement
        # recurse through the Qt selectors and can freeze Slicer 5.12.
        self.selectParameterNode(bindToWidget=False)
        self.segmentEditorWidget.setEffectNameOrder([])

    def displayLabelOfSegmentation(self):
        # Configure the parameter node before attaching it to the Qt widget.
        # Attaching an empty node first and then changing its references caused
        # recursive widget updates and froze Slicer 5.12.
        segmentEditorNode = self.selectParameterNode(bindToWidget=False)
        segmentationNode = self._currentSegmentationNode()
        if segmentationNode is None:
            raise RuntimeError("Segmentation node was not created")
        # Bind MRML nodes directly. Changing the embedded Qt node selector
        # recursively invoked its currentNodeChanged callback in Slicer 5.12
        # and froze the main thread on large segmentations.
        segmentEditorNode.SetAndObserveSegmentationNode(segmentationNode)
        sourceVolumeNode = getattr(self.logic, "_lastSourceVolumeNode", None)
        if sourceVolumeNode is not None and sourceVolumeNode.GetScene() is None:
            sourceVolumeNode = None
        if sourceVolumeNode is not None:
            segmentEditorNode.SetAndObserveSourceVolumeNode(sourceVolumeNode)
        else:
            pass
        self.segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
        self._parameterNode = segmentEditorNode
        self.segmentEditorWidget.updateWidgetFromMRML()
        return segmentEditorNode

    def selectParameterNode(self, bindToWidget=True):
        # Select parameter set node if one is found in the scene, and create one otherwise
        segmentEditorSingletonTag = "SegmentEditor"
        segmentEditorNode = slicer.mrmlScene.GetSingletonNode(segmentEditorSingletonTag, "vtkMRMLSegmentEditorNode")
        if segmentEditorNode is None:
            segmentEditorNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLSegmentEditorNode")
            segmentEditorNode.UnRegister(None)
            segmentEditorNode.SetSingletonTag(segmentEditorSingletonTag)
            segmentEditorNode = slicer.mrmlScene.AddNode(segmentEditorNode)
        if bindToWidget:
            self.segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
            self._parameterNode = segmentEditorNode
        return segmentEditorNode

    def getDefaultSourceVolumeNodeID(self):
        layoutManager = slicer.app.layoutManager()
        firstForegroundVolumeID = None
        # Use first background volume node in any of the displayed layouts.
        # If no beackground volume node is in any slice view then use the first
        # foreground volume node.
        for sliceViewName in layoutManager.sliceViewNames():
            sliceWidget = layoutManager.sliceWidget(sliceViewName)
            if not sliceWidget:
                continue
            compositeNode = sliceWidget.mrmlSliceCompositeNode()
            if compositeNode.GetBackgroundVolumeID():
                return compositeNode.GetBackgroundVolumeID()
            if compositeNode.GetForegroundVolumeID() and not firstForegroundVolumeID:
                firstForegroundVolumeID = compositeNode.GetForegroundVolumeID()
        # No background volume was found, so use the foreground volume (if any was found)
        return firstForegroundVolumeID

    def getVersionName(self) -> str:
        if self.setOverwriteCurrentVersion():
            return self.getCurrentLabelVersion()
        else:
            return self.currentImageData.getNewVersionName()

    def persistEditedSegmentation(self, newVersionName: str):
        self.currentImageData.updateSegmentationMetaByVerionTag(
            tag=newVersionName,
            status=self.getCurrentMetaStatus(),
            level=self.getCurrentMetaLevel(),
            approvedBy=self.selectedReviewer,
            comment=self.getCurrentComment(),
        )

        segmentationNode = None
        try:
            segmentationNode = self._currentSegmentationNode() or self.segmentEditorWidget.segmentationNode()
        except Exception:
            logging.exception("Embedded Segment Editor has no active segmentation")
        if segmentationNode is None:
            segmentationNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode")
        if segmentationNode is None:
            raise RuntimeError("Не найдена сегментация для сохранения")
        self.tmpdir = slicer.util.tempDirectory("slicer-monai-reviewer")
        label_in = tempfile.NamedTemporaryFile(suffix=".nrrd", dir=self.tmpdir).name
        slicer.util.saveNode(segmentationNode, label_in)

        self.logic.saveLabelInMonaiServer(imageData=self.currentImageData, label_in=label_in, tag=newVersionName)

    def deleteLabelByVersionTag(self):
        imageVersionTag = self.getCurrentLabelVersion()
        if self.isBlank(imageVersionTag):
            return
        if self.currentImageData.hasVersionTag(versionTag=imageVersionTag) is False:
            return
        self.logic.deleteLabelByVersionTag(imageData=self.currentImageData, versionTag=imageVersionTag)

    def updateAfterEditingSegmentation(self):
        imageVersionTag = self.getCurrentLabelVersion()
        setToSave = bool(self.ui.btn_save_new_version.isChecked())
        setToOverwrite = bool(self.ui.btn_overwrite_version.isChecked())
        setToDelete = bool(self.ui.btn_delete_version.isChecked())
        newLabelNameCreated = ""
        if setToSave:
            newLabelNameCreated = self.currentImageData.getNewVersionName()
            self.persistEditedSegmentation(newVersionName=newLabelNameCreated)

        elif setToOverwrite:
            if (imageVersionTag == self.LABEL.FINAL) or (imageVersionTag == self.LABEL.ORIGINAL):
                warningMessage: str = (
                    "Initial Segmentation with label 'final' or 'original' \ncannot be overwritten.\n Please save current edit as new version."
                )
                slicer.util.warningDisplay(warningMessage)
                logging.warn(warningMessage)
                return

            self.persistEditedSegmentation(newVersionName=imageVersionTag)

        elif setToDelete:
            if (imageVersionTag == self.LABEL.FINAL) or (imageVersionTag == self.LABEL.ORIGINAL):
                warningMessage: str = "Initial Segmentation with label 'final' or 'original' \ncannot be deleted."
                slicer.util.warningDisplay(warningMessage)
                logging.warn(warningMessage)
                return

            self.deleteLabelByVersionTag()

        self.resetSegmentationEditorTools()
        self.reloadImageAfterEditingLabel()
        if newLabelNameCreated != "":
            self.setVersionTagInComboBox(versionTag=newLabelNameCreated)

    def reloadImageAfterEditingLabel(self):
        imageId = self.currentImageData.getFileName()
        latestVersion = self.currentImageData.getLatestVersionTag()
        logging.info(f"{self.getCurrentTime()}: Loading image (id='{imageId}') with version tag = '{latestVersion}'")
        self.loadNextImage(imageData=self.currentImageData, tag=latestVersion)
        self.fillComboBoxLabelVersions(self.currentImageData)

    def processDataStoreRecords(self):
        serverUrl: str = self.ui.comboBox_server_url.currentText
        result: bool = self.logic.initMetaDataProcessing()
        if result is False:
            warningMessage = (
                "Request for datastore-info failed.\nPlease check if server address is correct \n('{}')!".format(
                    serverUrl
                )
            )
            slicer.util.warningDisplay(warningMessage)
            logging.warn(warningMessage)
            return
        logging.info(f"{self.getCurrentTime()}: Successfully processed all records in datastore.")

    def reloadOverallStatistic(self):
        if self.reviewersModeIsActive:
            self.processDataStoreRecords()
            self.setProgessBar()
            self.setProgressBarOfAll()

    def displayAdditionalMetaIfEdited(self, tag: str):
        if tag == self.LABEL.FINAL or tag == self.LABEL.ORIGINAL:
            self.ui.lineEdit_editor.hide()
            self.ui.lineEdit_editing_date.hide()
            self.ui.label_editor.hide()
            self.ui.label_editing_date.hide()
        else:
            self.ui.lineEdit_editor.show()
            self.ui.lineEdit_editing_date.show()
            self.ui.label_editor.show()
            self.ui.label_editing_date.show()

    def isBlank(self, string) -> bool:
        return not (string and string.strip())


class MONAILabelReviewerLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)
        self.temp_dir = None
        self.imageDataController: ImageDataController = ImageDataController()
        self._lastSourceVolumeNode = None

    # Section: Server
    def getServerUrl(self) -> str:
        return self.imageDataController.getServerUrl()

    def getCurrentTime(self) -> datetime:
        return datetime.datetime.now()

    def connectToMonaiServer(self, serverUrl: str) -> bool:
        return self.imageDataController.connectToMonaiServer(serverUrl)

    def getMapIdToImageData(self) -> Dict[str, ImageData]:
        """
        Returns dictionary (Dict[str:ImageData]) which maps id to Imagedata-object
        """
        return self.imageDataController.getMapIdToImageData()

    def initMetaDataProcessing(self) -> bool:
        return self.imageDataController.initMetaDataProcessing()

    def getStatistics(self) -> ImageDataStatistics:
        return self.imageDataController.getStatistics()

    def getClientIds(self) -> List[str]:
        return self.imageDataController.getClientIds()

    def getReviewers(self) -> List[str]:
        return self.imageDataController.getReviewers()

    # Section: Loading images
    def getAllImageData(
        self, segmented: str, isNotSegmented: str, isApproved: bool, isFlagged: bool
    ) -> List[ImageData]:
        return self.imageDataController.getAllImageData(segmented, isNotSegmented, isApproved, isFlagged)

    def getImageDataByClientId(self, selectedClientId: str, isApproved: bool, isFlagged: bool) -> List[ImageData]:
        return self.imageDataController.getImageDataByClientId(selectedClientId, isApproved, isFlagged)

    def getPercentageApproved(self, selectedClientId: str):
        percentageApprovedOfClient, idxApprovedOfClient = self.imageDataController.getPercentageApproved(
            selectedClientId
        )
        return percentageApprovedOfClient, idxApprovedOfClient

    def getPercentageSemgmentedByClient(self, selectedClientId: str):
        percentageSemgmentedByClient, idxSegmentedByClient = self.imageDataController.getPercentageSemgmentedByClient(
            selectedClientId
        )
        return percentageSemgmentedByClient, idxSegmentedByClient

    # Section: Search Image
    def getMultImageDataByIds(self, idList: List[str]) -> Dict[str, ImageData]:
        return self.imageDataController.getMultImageDataByIds(idList)

    def searchByAnnotatorReviewer(
        self, selectedAnnotator: str, selectedReviewer: str, isApproved: bool, isFlagged: bool
    ) -> Dict[str, ImageData]:
        return self.imageDataController.searchByAnnotatorReviewer(
            selectedAnnotator, selectedReviewer, isApproved, isFlagged
        )

    def searchByLevel(self, isEasy: bool, isMedium: bool, isHard: bool) -> Dict[str, ImageData]:
        return self.imageDataController.getImageDataByLevel(isEasy=isEasy, isMedium=isMedium, isHard=isHard)

    def updateImageData(
        self, imageData: ImageData, versionTag: str, status: str, level: str, approvedBy: str, comment: str
    ) -> dict:
        """
        update meta data in information box
        Returns: jsonDict: json dictionary which contains updated meta data
        """
        imageId = imageData.getName()
        isEqual = imageData.isEqualSegmentationMeta(
            tag=versionTag, status=status, level=level, approvedBy=approvedBy, comment=comment
        )
        if isEqual:
            logging.info(f"{self.getCurrentTime()}: No changes for image (id='{imageId}')")
            return ""

        imageData.updateSegmentationMetaByVerionTag(
            tag=versionTag, status=status, level=level, approvedBy=approvedBy, comment=comment
        )
        jsonDict = imageData.getMetaByVersionTag(tag=versionTag)

        if jsonDict is None:
            logging.info(f"{self.getCurrentTime()}: No update for Image (id='{imageId}')")
            return ""
        logging.info(f"{self.getCurrentTime()}: Successfully updated Image (id='{imageId}')")
        return jsonDict

    # Section: Dicom stream
    def updateLabelInfo(
        self, imageData: ImageData, versionTag: str, status: str, level: str, approvedBy: str, comment: str
    ):
        imageId = imageData.getName()
        updatedMetaJson = self.updateImageData(imageData, versionTag, status, level, approvedBy, comment)
        if updatedMetaJson == "":
            logging.info(
                "{} :  Image update (id='{}', version tag='{}') is empty".format(
                    self.getCurrentTime(), imageId, versionTag
                )
            )
            return

        logging.info(f"{self.getCurrentTime()} : Image update (id='{imageId}', version tag='{versionTag}')")
        logging.info(updatedMetaJson)
        self.imageDataController.updateLabelInfoOfAllVersionTags(
            imageData=imageData, versionTag=versionTag, level=level, updatedMetaJson=updatedMetaJson
        )

    def loadDicomAndSegmentation(self, imageData: ImageData, tag: str):
        """
        Loads original Dicom image and Segmentation into slicer window
        Parameters:
          imageData (ImageData): Contains meta data (of dicom and segmenation)
                                 which is required for rest request to monai server
                                 in order to get dicom and segmenation (.nrrd).
        """
        # Request dicom
        image_name = imageData.getFileName()
        image_id = imageData.getName()
        node_name = imageData.getNodeName()
        logging.info(
            "{}: Request Data  image_name='{}', node_name='{}', image_id='{}'".format(
                self.getCurrentTime(), image_name, node_name, image_id
            )
        )

        self.requestDicomImage(image_id, image_name, node_name)
        self.setTempFolderDir()

        # Request segmentation
        if imageData.isSegemented():
            segmentationFileName = imageData.getSegmentationFileName()
            img_blob = self.imageDataController.reuqestSegmentation(image_id, tag)
            if img_blob is None or getattr(img_blob, "status_code", None) != 200:
                logging.warning(
                    "Mask check '%s' tag=%s: HTTP %s - no mask imported, MRI kept",
                    image_id, tag, getattr(img_blob, "status_code", None),
                )
                return
            destination = self.storeSegmentation(img_blob, segmentationFileName, self.temp_dir.name)
            if destination and os.path.isfile(destination):
                nonzero, dims = self.countNrrdNonzeroVoxels(destination)
                logging.info(
                    "Mask check '%s' tag=%s: HTTP %s, %s bytes, NRRD dims=%s, nonzero voxels=%s",
                    image_id, tag, img_blob.status_code, len(img_blob.content), dims, nonzero,
                )
                if nonzero == 0:
                    logging.warning("Mask '%s' tag=%s has no nonzero voxels on the server", image_id, tag)
                segmentationNode = self.displaySegmention(destination)
                os.remove(destination)
                summary = self.summarizeSegmentation(segmentationNode)
                if summary:
                    logging.info(
                        "Segmentation '%s': %s segments, %s non-empty, labelmap dims=%s, scalar range=%s",
                        image_id, summary["segments"], summary["nonEmptySegments"],
                        summary["labelmapDims"], summary["scalarRange"],
                    )
        else:
            logging.warning("%s: datastore entry has no usable label version for tag=%s", self.getCurrentTime(), tag)

    def storeSegmentation(
        self, response: requests.models.Response, segmentationFileName: str, tempDirectory: str
    ) -> str:
        """
        stores loaded segmenation temporarily in local directory
        Parameters:
            response (requests.models.Response): contains segmentation data
            image_id (str): image id of segmentation
        """
        segmentation = response.content
        destination = self.getPathToStore(segmentationFileName, tempDirectory)
        with open(destination, "wb") as img_file:
            img_file.write(segmentation)
        logging.info(f"{self.getCurrentTime()}: Image segmentation is stored temoparily in: {destination}")
        return destination

    def countNrrdNonzeroVoxels(self, path: str):
        """Return (nonzero voxel count, dims) of a downloaded label file, or (None, None)."""
        try:
            import numpy as np
            import SimpleITK as sitk

            image = sitk.ReadImage(path)
            array = sitk.GetArrayViewFromImage(image)
            return int(np.count_nonzero(array)), tuple(image.GetSize())
        except Exception:
            logging.exception("Could not read downloaded mask for voxel check")
            return None, None

    @staticmethod
    def segmentVoxelCount(segment, binaryName=None):
        """Number of voxels that really belong to the segment (shared labelmaps aware)."""
        import numpy as np
        from vtk.util import numpy_support

        if binaryName is None:
            binaryName = slicer.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName()
        representation = segment.GetRepresentation(binaryName) if segment is not None else None
        if representation is None or representation.GetPointData() is None:
            return 0
        dims = representation.GetDimensions()
        if min(dims) <= 0:
            return 0
        scalars = representation.GetPointData().GetScalars()
        if scalars is None:
            return 0
        array = numpy_support.vtk_to_numpy(scalars)
        labelValue = segment.GetLabelValue() if hasattr(segment, "GetLabelValue") else 1
        return int(np.count_nonzero(array == labelValue))

    @staticmethod
    def isUsableLabelmap(imageData):
        """Point 7: reject empty resampling results."""
        if imageData is None:
            return False, "no image"
        dims = imageData.GetDimensions()
        if min(dims) <= 0:
            return False, f"zero dims {dims}"
        scalarRange = imageData.GetScalarRange()
        if scalarRange[0] == 0.0 and scalarRange[1] == 0.0:
            return False, "scalar range (0.0, 0.0)"
        import numpy as np
        from vtk.util import numpy_support

        scalars = imageData.GetPointData().GetScalars() if imageData.GetPointData() else None
        if scalars is None or np.count_nonzero(numpy_support.vtk_to_numpy(scalars)) == 0:
            return False, "no nonzero voxels"
        return True, ""

    def summarizeSegmentation(self, segmentationNode):
        """Real voxel-based summary (segment names alone prove nothing)."""
        if segmentationNode is None or segmentationNode.GetScene() is None:
            return None
        binaryName = slicer.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName()
        segmentation = segmentationNode.GetSegmentation()
        perSegment = {}
        dims = None
        scalarRange = None
        for index in range(segmentation.GetNumberOfSegments()):
            segment = segmentation.GetNthSegment(index)
            perSegment[segment.GetName()] = self.segmentVoxelCount(segment, binaryName)
            representation = segment.GetRepresentation(binaryName)
            if representation is not None and perSegment[segment.GetName()] > 0 and dims is None:
                dims = tuple(representation.GetDimensions())
                scalarRange = tuple(representation.GetScalarRange())
        return {
            "segments": segmentation.GetNumberOfSegments(),
            "nonEmptySegments": sum(1 for count in perSegment.values() if count > 0),
            "voxelsPerSegment": perSegment,
            "labelmapDims": dims,
            "scalarRange": scalarRange,
        }

    def getPathToStore(self, segmentationFileName: str, tempDirectory: str) -> str:
        return tempDirectory + "/" + segmentationFileName

    def displaySegmention(self, destination: str):
        """
        Displays the segmentation in slicer window
        """
        segmentation = slicer.util.loadSegmentation(destination)
        if segmentation is None:
            logging.warning("%s: Segmentation node was not created", self.getCurrentTime())
            return None
        sourceVolumeNode = self._lastSourceVolumeNode
        if sourceVolumeNode is not None and sourceVolumeNode.GetScene() is None:
            sourceVolumeNode = None
        if sourceVolumeNode is not None:
            self.alignSegmentationToSourceVolume(segmentation, sourceVolumeNode)
        else:
            logging.warning("Current series MRI is unknown; mask kept in its own geometry (not aligned)")
        self._configureSegmentationDisplay(segmentation)
        self._lastSegmentationNode = segmentation
        return segmentation

    def alignSegmentationToSourceVolume(self, segmentationNode, sourceVolumeNode):
        """Resample loaded labelmaps into the MRI RAS/IJK geometry in memory.

        Some MONAI datastore labelmaps use a different axis order/sign
        convention than the NIfTI MRI.  They can therefore appear in 3D but
        miss every 2D slice.  Exporting each segment to the source-volume
        reference geometry with nearest-neighbor interpolation preserves the
        discrete labels and does not modify the server response or datastore.
        """
        if segmentationNode is None or sourceVolumeNode is None:
            return False

        self._fixMirroredLpsSegmentation(segmentationNode, sourceVolumeNode)
        segmentation = segmentationNode.GetSegmentation()
        binaryName = slicer.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName()
        segmentInfos = []
        segmentIds = vtk.vtkStringArray()
        segmentation.GetSegmentIDs(segmentIds)
        for index in range(segmentIds.GetNumberOfValues()):
            segmentId = segmentIds.GetValue(index)
            segment = segmentation.GetSegment(segmentId)
            hasVoxels = self.segmentVoxelCount(segment, binaryName) > 0
            segmentInfos.append((segmentId, segment.GetName(), segment.GetColor(), hasVoxels))

        nonEmpty = [info for info in segmentInfos if info[3]]
        if not nonEmpty:
            segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(sourceVolumeNode)
            return False

        logic = slicer.modules.segmentations.logic()
        alignedSegments = []
        for segmentId, segmentName, segmentColor, hasVoxels in nonEmpty:
            oneSegmentIds = vtk.vtkStringArray()
            oneSegmentIds.InsertNextValue(segmentId)
            labelmapNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
            try:
                exported = logic.ExportSegmentsToLabelmapNode(
                    segmentationNode,
                    oneSegmentIds,
                    labelmapNode,
                    sourceVolumeNode,
                    slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY,
                )
                if not exported or labelmapNode.GetImageData() is None:
                    logging.warning("Could not align segment '%s' to MRI geometry; preserving original segment", segmentName)
                    continue
                usable, reason = self.isUsableLabelmap(labelmapNode.GetImageData())
                if not usable:
                    logging.warning(
                        "Alignment result for '%s' rejected (%s); preserving original segment",
                        segmentName, reason,
                    )
                    continue

                alignedImage = slicer.vtkOrientedImageData()
                alignedImage.DeepCopy(labelmapNode.GetImageData())
                imageToWorld = vtk.vtkMatrix4x4()
                labelmapNode.GetIJKToRASMatrix(imageToWorld)
                alignedImage.SetImageToWorldMatrix(imageToWorld)

                alignedSegment = vtkSegmentationCore.vtkSegment()
                alignedSegment.SetName(segmentName)
                alignedSegment.SetColor(*segmentColor)
                alignedSegment.AddRepresentation(binaryName, alignedImage)
                alignedSegments.append((segmentId, alignedSegment))
            finally:
                slicer.mrmlScene.RemoveNode(labelmapNode)

        if not alignedSegments:
            logging.warning("No usable alignment result; original mask kept unchanged")
            return False

        # Replace only segments whose aligned result is valid; every other
        # segment (including failed/empty alignments) stays untouched.
        for segmentId, alignedSegment in alignedSegments:
            orderedIds = list(segmentation.GetSegmentIDs())
            position = orderedIds.index(segmentId)
            insertBefore = orderedIds[position + 1] if position + 1 < len(orderedIds) else ""
            segmentation.RemoveSegment(segmentId)
            segmentation.AddSegment(alignedSegment, segmentId, insertBefore)

        segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(sourceVolumeNode)
        logging.info(
            "Aligned %s non-empty segments to source MRI geometry (%s)",
            len(alignedSegments),
            sourceVolumeNode.GetName(),
        )
        return True

    def _fixMirroredLpsSegmentation(self, segmentationNode, sourceVolumeNode):
        """Labels saved in LPS but read as RAS land mirrored in X and Y.

        If the segmentation RAS bounds equal the source volume bounds with
        X and Y negated, harden a diag(-1,-1,1) transform into it.
        """
        try:
            vb = [0.0] * 6
            sb = [0.0] * 6
            sourceVolumeNode.GetRASBounds(vb)
            segmentationNode.GetRASBounds(sb)
            if sb[0] > sb[1]:
                return False

            def close(a, b):
                return abs(a - b) <= max(2.0, 0.02 * (abs(a) + abs(b)))

            overlapsX = sb[0] < vb[1] and sb[1] > vb[0]
            overlapsY = sb[2] < vb[3] and sb[3] > vb[2]
            mirrored = (
                close(sb[0], -vb[1]) and close(sb[1], -vb[0])
                and close(sb[2], -vb[3]) and close(sb[3], -vb[2])
            )
            if not mirrored or (overlapsX and overlapsY and close(sb[0], vb[0])):
                return False
            flip = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", "MONAI_LPS_to_RAS")
            matrix = vtk.vtkMatrix4x4()
            matrix.SetElement(0, 0, -1)
            matrix.SetElement(1, 1, -1)
            flip.SetMatrixTransformToParent(matrix)
            segmentationNode.SetAndObserveTransformNodeID(flip.GetID())
            slicer.vtkSlicerTransformLogic().hardenTransform(segmentationNode)
            slicer.mrmlScene.RemoveNode(flip)
            return True
        except Exception:
            logging.exception("LPS mirror check failed")
            return False
    def _configureSegmentationDisplay(self, segmentation):
        """Enable binary-labelmap overlays and per-segment visibility."""
        if segmentation is None:
            return
        try:
            if hasattr(segmentation, "CreateBinaryLabelmapRepresentation"):
                segmentation.CreateBinaryLabelmapRepresentation()
        except Exception:
            logging.exception("Could not create binary labelmap representation")

        if segmentation.GetDisplayNode() is None:
            segmentation.CreateDefaultDisplayNodes()
        displayNode = segmentation.GetDisplayNode()
        if displayNode is None:
            return

        displayNode.SetVisibility(True)
        if hasattr(displayNode, "SetVisibility2D"):
            displayNode.SetVisibility2D(True)
        if hasattr(displayNode, "SetVisibility2DFill"):
            displayNode.SetVisibility2DFill(True)
        if hasattr(displayNode, "SetVisibility2DOutline"):
            displayNode.SetVisibility2DOutline(True)
        if hasattr(displayNode, "SetOpacity2DFill"):
            displayNode.SetOpacity2DFill(0.5)
        if hasattr(displayNode, "SetOpacity2DOutline"):
            displayNode.SetOpacity2DOutline(1.0)
        if hasattr(displayNode, "SetVisibility3D"):
            displayNode.SetVisibility3D(True)

        # After MONAI Reviewer clears the scene, Slicer 5.12 may leave the
        # segmentation display node without explicit slice-view bindings.
        # Bind it to every current slice view so 2D overlays are reliable.
        if hasattr(displayNode, "RemoveAllViewNodeIDs"):
            displayNode.RemoveAllViewNodeIDs()
        for sliceNode in slicer.util.getNodesByClass("vtkMRMLSliceNode") + slicer.util.getNodesByClass("vtkMRMLViewNode"):
            if hasattr(displayNode, "AddViewNodeID"):
                displayNode.AddViewNodeID(sliceNode.GetID())

        segmentIds = vtk.vtkStringArray()
        segmentation.GetSegmentation().GetSegmentIDs(segmentIds)
        for index in range(segmentIds.GetNumberOfValues()):
            segmentId = segmentIds.GetValue(index)
            displayNode.SetSegmentVisibility(segmentId, True)
            if hasattr(displayNode, "SetSegmentVisibility2DFill"):
                displayNode.SetSegmentVisibility2DFill(segmentId, True)
            if hasattr(displayNode, "SetSegmentVisibility2DOutline"):
                displayNode.SetSegmentVisibility2DOutline(segmentId, True)
            if hasattr(displayNode, "SetSegmentVisibility3D"):
                displayNode.SetSegmentVisibility3D(segmentId, True)
        logging.debug("Segmentation visibility enabled for %s segments", segmentIds.GetNumberOfValues())

    def buildClosedSurface(self, segmentationNode):
        """Separate, explicit 3D operation. Returns build time in seconds or None."""
        if segmentationNode is None or segmentationNode.GetScene() is None:
            return None
        closedName = slicer.vtkSegmentationConverter.GetSegmentationClosedSurfaceRepresentationName()
        started = time.monotonic()
        try:
            slicer.app.setOverrideCursor(qt.Qt.BusyCursor)
            if not segmentationNode.GetSegmentation().ContainsRepresentation(closedName):
                segmentationNode.CreateClosedSurfaceRepresentation()
            display = segmentationNode.GetDisplayNode()
            if display is not None:
                display.SetVisibility3D(True)
                # Do not change the preferred representation here. In Slicer
                # 5.12 that setter can recursively trigger Qt display updates
                # and cause c00000fd (stack overflow) in Qt5Core.dll.
        except Exception:
            logging.exception("Could not build 3D surface")
            return None
        finally:
            slicer.app.restoreOverrideCursor()
        elapsed = time.monotonic() - started
        logging.info("3D surfaces built in %.1f s", elapsed)
        return elapsed

    def ensureAllSegmentationViewsVisible(self):
        """Keep loaded labels visible in both 2D slices and the 3D view."""
        for segmentation in slicer.util.getNodesByClass("vtkMRMLSegmentationNode"):
            self._configureSegmentationDisplay(segmentation)

    def ensureProjectSegmentSchema(self):
        """Create the complete Task04/Task05 segment list without changing masks."""
        serverUrl = self.getServerUrl().rstrip("/").lower()
        fullSchemaRgb = [
            ("bone_humerus", (255, 214, 10)),
            ("bone_scapula", (255, 160, 20)),
            ("bone_clavicle", (255, 105, 40)),
            ("muscle_supraspinatus", (45, 105, 210)),
            ("muscle_infraspinatus", (46, 160, 90)),
            ("muscle_subscapularis", (205, 55, 75)),
            ("muscle_deltoid", (40, 190, 210)),
            ("muscle_biceps_brachii", (130, 80, 190)),
            ("muscle_brachialis", (230, 190, 40)),
            ("muscle_coracobrachialis", (190, 70, 50)),
            ("muscle_latissimus_dorsi", (70, 180, 150)),
            ("muscle_pectoralis_major", (220, 100, 180)),
            ("muscle_teres_major", (100, 130, 220)),
            ("muscle_teres_minor", (180, 140, 60)),
            ("muscle_triceps_brachii", (80, 190, 190)),
        ]
        if serverUrl.endswith(":8103") or "/monai/task04" in serverUrl:
            projectName = "Task04"
            schemaRgb = fullSchemaRgb[:7]
        elif serverUrl.endswith(":8104") or "/monai/task05" in serverUrl:
            projectName = "Task05"
            schemaRgb = fullSchemaRgb
        else:
            return None
        schema = [(name, tuple(channel / 255.0 for channel in rgb)) for name, rgb in schemaRgb]

        segmentationNodes = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
        lastNode = getattr(self, "_lastSegmentationNode", None)
        if lastNode is not None and lastNode.GetScene() is not None:
            segmentationNode = lastNode
        elif segmentationNodes:
            segmentationNode = segmentationNodes[-1]
        else:
            segmentationNode = slicer.mrmlScene.AddNewNode("vtkMRMLSegmentationNode", f"{projectName}_Segmentation")

        sourceVolumeNode = getattr(self, "_lastSourceVolumeNode", None)
        if sourceVolumeNode is not None and sourceVolumeNode.GetScene() is None:
            sourceVolumeNode = None
        if sourceVolumeNode is not None and hasattr(segmentationNode, "SetReferenceImageGeometryParameterFromVolumeNode"):
            segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(sourceVolumeNode)

        segmentation = segmentationNode.GetSegmentation()
        # Names follow the server /info labels (label value = position + 1).
        # Older local builds used names without the "muscle_" prefix; such
        # segments are matched and renamed instead of creating duplicates.
        existingSegmentsByName = {}
        for index in range(segmentation.GetNumberOfSegments()):
            segment = segmentation.GetNthSegment(index)
            existingSegmentsByName.setdefault(segment.GetName(), segment)
        for name, _color in schema:
            legacyName = name[len("muscle_"):] if name.startswith("muscle_") else None
            if name not in existingSegmentsByName and legacyName in existingSegmentsByName:
                segment = existingSegmentsByName.pop(legacyName)
                segment.SetName(name)
                existingSegmentsByName[name] = segment

        added = 0
        for name, color in schema:
            segment = existingSegmentsByName.get(name)
            if segment is None:
                # In the Slicer 5.12 Python wrapper AddEmptySegment(name)
                # may interpret the argument as an ID and still create a
                # generic display name.  Create the segment explicitly so
                # the Segment Editor receives the project name and color.
                segment = vtkSegmentationCore.vtkSegment()
                segment.SetName(name)
                segment.SetColor(*color)
                segmentation.AddSegment(segment)
                added += 1
            if segment is not None:
                segment.SetColor(*color)

        self._configureSegmentationDisplay(segmentationNode)
        self._lastSegmentationNode = segmentationNode
        logging.info("%s schema ensured: %s segments total, %s empty segments added", projectName, len(schema), added)
        return segmentationNode

    def ensureTask04SegmentSchema(self):
        """Compatibility alias for earlier local Reviewer builds."""
        return self.ensureProjectSegmentSchema()

    def requestDicomImage(self, image_id: str, image_name: str, node_name: str):
        download_uri = self.imageDataController.getDicomDownloadUri(image_id)
        self._lastSourceVolumeNode = None
        beforeIds = {node.GetID() for node in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")}
        loaded = SampleData.SampleDataLogic().downloadFromURL(nodeNames=node_name, fileNames=image_name, uris=download_uri)

        # Always keep the volume that belongs to the just requested series.
        # Using GetFirstNodeByClass() is unsafe after reviewing more than one
        # series because the scene may still contain an older scalar volume.
        candidates = []
        if loaded is not None:
            candidates = loaded if isinstance(loaded, (list, tuple)) else [loaded]
        candidates += [
            node for node in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
            if node.GetID() not in beforeIds
        ]
        for node in candidates:
            if node is not None and hasattr(node, "GetID"):
                self._lastSourceVolumeNode = node
                logging.info("Selected current MRI volume for mask alignment: %s", node.GetName())
                return

        for node in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"):
            if node.GetName() in (node_name, image_name) or node.GetName().startswith(node_name):
                self._lastSourceVolumeNode = node
                logging.info("Selected current MRI volume by name for mask alignment: %s", node.GetName())
                return

        self._lastSourceVolumeNode = None
        logging.warning("Could not identify the MRI of the current series; mask will not be resampled")

    def setTempFolderDir(self):
        """
        Create temporary dirctory to store the downloaded segmentation (.nrrd)
        """
        if self.temp_dir is None:
            self.temp_dir = tempfile.TemporaryDirectory()
        logging.info(f"{self.getCurrentTime()}: Temporary Directory: '{self.temp_dir.name}'")

    def saveLabelInMonaiServer(self, imageData: ImageData, label_in: str, tag: str):
        imageName = imageData.getName()
        params = imageData.obtainUpdatedParams(tag)
        self.imageDataController.saveLabelInMonaiServer(imageName, label_in, tag, params)

    def deleteLabelByVersionTag(self, imageData: ImageData, versionTag: str) -> bool:
        imageId = imageData.getName()
        imageData.deleteVersionName(versionTag)
        successfullyDeleted: bool = self.imageDataController.deleteLabelByVersionTag(imageId, versionTag)
        return successfullyDeleted


class MONAILabelReviewerTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_MONAILabelReviewer1()

    def test_MONAILabelReviewer1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")
