import React, { useState } from 'react';
import {
  Container,
  Paper,
  TextField,
  Button,
  Typography,
  Box,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import { useAuth } from '../hooks/useAuth';
import { useMedicalHistory } from '../hooks/useMedicalHistory';
import AddIcon from '@mui/icons-material/Add';
import EditIcon from '@mui/icons-material/Edit';
import DeleteIcon from '@mui/icons-material/Delete';

const MedicalHistory = () => {
  const { user } = useAuth();
  const {
    medicalRecords,
    addRecord,
    updateRecord,
    deleteRecord,
    loading,
    error,
  } = useMedicalHistory();
  const [open, setOpen] = useState(false);
  const [recordType, setRecordType] = useState('');
  const [recordContent, setRecordContent] = useState('');
  const [editingRecord, setEditingRecord] = useState(null);

  const handleAddRecord = async () => {
    if (!recordType || !recordContent) return;
    try {
      await addRecord({
        type: recordType,
        content: recordContent,
        userId: user.id,
      });
      setOpen(false);
      setRecordType('');
      setRecordContent('');
    } catch (err) {
      console.error('Error adding record:', err);
    }
  };

  const handleEditRecord = async (record) => {
    setEditingRecord(record);
    setRecordType(record.type);
    setRecordContent(record.content);
    setOpen(true);
  };

  const handleUpdateRecord = async () => {
    if (!editingRecord || !recordType || !recordContent) return;
    try {
      await updateRecord({
        id: editingRecord.id,
        type: recordType,
        content: recordContent,
      });
      setOpen(false);
      setEditingRecord(null);
      setRecordType('');
      setRecordContent('');
    } catch (err) {
      console.error('Error updating record:', err);
    }
  };

  const handleDeleteRecord = async (record) => {
    try {
      await deleteRecord(record.id);
    } catch (err) {
      console.error('Error deleting record:', err);
    }
  };

  return (
    <Container component="main" maxWidth="md">
      <Paper elevation={3} sx={{ p: 4, mt: 8 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 4 }}>
          <Typography variant="h5">Medical History</Typography>
          <Button
            variant="contained"
            color="primary"
            startIcon={<AddIcon />}
            onClick={() => {
              setOpen(true);
              setEditingRecord(null);
              setRecordType('');
              setRecordContent('');
            }}
          >
            Add Record
          </Button>
        </Box>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Type</TableCell>
                <TableCell>Content</TableCell>
                <TableCell>Date</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {medicalRecords.map((record) => (
                <TableRow key={record.id}>
                  <TableCell>{record.type}</TableCell>
                  <TableCell>{record.content}</TableCell>
                  <TableCell>
                    {new Date(record.createdAt).toLocaleDateString()}
                  </TableCell>
                  <TableCell>
                    <IconButton
                      onClick={() => handleEditRecord(record)}
                      size="small"
                    >
                      <EditIcon />
                    </IconButton>
                    <IconButton
                      onClick={() => handleDeleteRecord(record)}
                      size="small"
                      color="error"
                    >
                      <DeleteIcon />
                    </IconButton>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>

      <Dialog open={open} onClose={() => setOpen(false)}>
        <DialogTitle>
          {editingRecord ? 'Edit Medical Record' : 'Add Medical Record'}
        </DialogTitle>
        <DialogContent>
          <Box sx={{ mt: 2 }}>
            <TextField
              fullWidth
              label="Record Type"
              value={recordType}
              onChange={(e) => setRecordType(e.target.value)}
              sx={{ mb: 2 }}
            />
            <TextField
              fullWidth
              label="Record Content"
              multiline
              rows={4}
              value={recordContent}
              onChange={(e) => setRecordContent(e.target.value)}
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpen(false)}>Cancel</Button>
          <Button
            variant="contained"
            color="primary"
            onClick={editingRecord ? handleUpdateRecord : handleAddRecord}
          >
            {editingRecord ? 'Update' : 'Add'}
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default MedicalHistory;
