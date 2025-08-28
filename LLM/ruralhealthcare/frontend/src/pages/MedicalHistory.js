import React, { useState } from 'react';
import {
  Container,
  Paper,
  Typography,
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import { useAuth } from '../hooks/useAuth';
import { useMedicalHistory } from '../hooks/useMedicalHistory';
import { useTranslation } from 'react-i18next';

const MedicalHistory = () => {
  const { user } = useAuth();
  const { medicalRecords, addRecord, deleteRecord } = useMedicalHistory();
  const [open, setOpen] = React.useState(false);
  const [newRecord, setNewRecord] = React.useState({
    condition: '',
    date: '',
    notes: '',
  });
  const [viewOpen, setViewOpen] = useState(false);
  const [selectedChat, setSelectedChat] = useState('');
  const [selectedRecord, setSelectedRecord] = useState(null);
  const [file, setFile] = useState(null);
  const { t } = useTranslation();

  console.log('MedicalHistory component rendered');

  const handleAddRecord = async () => {
    const formData = new FormData();
    formData.append('condition', newRecord.condition);
    formData.append('date', newRecord.date);
    formData.append('notes', newRecord.notes);
    if (file) formData.append('file', file);
    formData.append('user_id', user.id);
    await addRecord(formData, true);
    setOpen(false);
    setNewRecord({ condition: '', date: '', notes: '' });
    setFile(null);
  };

  const handleDeleteRecord = async (id) => {
    await deleteRecord(id);
  };

  // Defensive: always use an array
  const records = Array.isArray(medicalRecords) ? medicalRecords : [];

  return (
    <Container component="main" maxWidth="lg">
      <Paper elevation={3} sx={{ p: 4, mt: 8 }}>
        <Box sx={{ mb: 4 }}>
          <Typography variant="h5" component="h1" gutterBottom>
            {t('medical_history')}
          </Typography>
          <Typography variant="subtitle1" color="textSecondary">
            {t('patient')}: {user?.name}
          </Typography>
        </Box>

        <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 3 }}>
          <Button
            variant="contained"
            color="primary"
            onClick={() => setOpen(true)}
          >
            {t('add_medical_record')}
          </Button>
        </Box>

        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>{t('date')}</TableCell>
                <TableCell>{t('condition')}</TableCell>
                <TableCell>{t('notes')}</TableCell>
                <TableCell>{t('actions')}</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {records.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={4} align="center">{t('no_records')}</TableCell>
                </TableRow>
              ) : (
                records.map((record) => (
                  <TableRow key={record.id}>
                    <TableCell>{new Date(record.date).toLocaleDateString()}</TableCell>
                    <TableCell>{record.condition}</TableCell>
                    <TableCell>
                      {record.notes}
                      {record.report_file && (
                        <Box sx={{ mt: 1 }}>
                          <a
                            href={record.report_file.startsWith('http') ? record.report_file : `http://localhost:8000/${record.report_file}`}
                            target="_blank"
                            rel="noopener noreferrer"
                          >
                            📎 {t('view_report')}
                          </a>
                        </Box>
                      )}
                    </TableCell>
                    <TableCell>
                      <Button
                        variant="outlined"
                        color="primary"
                        size="small"
                        onClick={() => {
                          setSelectedChat(record.symptoms || '');
                          setSelectedRecord(record);
                          setViewOpen(true);
                        }}
                        sx={{ mr: 1 }}
                      >
                        {t('view')}
                      </Button>
                      <Button
                        variant="outlined"
                        color="error"
                        size="small"
                        onClick={() => handleDeleteRecord(record.id)}
                      >
                        {t('delete')}
                      </Button>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>

      {/* View Chat Dialog */}
      <Dialog open={viewOpen} onClose={() => setViewOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>{t('chat_session_details')}</DialogTitle>
        <DialogContent>
          <Box sx={{ p: 2 }}>
            {/* Show uploaded file link if present */}
            {selectedRecord && selectedRecord.report_file && (
              <Box sx={{ mb: 2 }}>
                <a
                  href={selectedRecord.report_file.startsWith('http') ? selectedRecord.report_file : `http://localhost:8000/${selectedRecord.report_file}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{ fontWeight: 'bold', color: '#1976d2' }}
                >
                  📎 {t('view_report')}
                </a>
              </Box>
            )}

            {/* Show chat session if present, else show a message */}
            {selectedChat && selectedChat.trim() ? (
              selectedChat
                .split('\n')
                .filter(line => line.trim() !== '')
                .map((line, idx) => {
                  let isUser = line.trim().startsWith('user:');
                  let isBot = line.trim().startsWith('bot:');
                  let text = line.replace(/^user:\s*/i, '').replace(/^bot:\s*/i, '');

                  return (
                    <Box
                      key={idx}
                      sx={{
                        display: 'flex',
                        justifyContent: isUser ? 'flex-end' : 'flex-start',
                        mb: 1,
                      }}
                    >
                      <Box
                        sx={{
                          bgcolor: isUser ? '#e3f2fd' : '#f5f5f5',
                          color: 'black',
                          px: 2,
                          py: 1,
                          borderRadius: 2,
                          maxWidth: '80%',
                          boxShadow: 1,
                          fontFamily: 'inherit',
                          textAlign: 'left',
                        }}
                      >
                        <b style={{ color: isUser ? '#1976d2' : '#388e3c' }}>
                          {isUser ? t('you') : isBot ? t('ai_doctor') : ''}
                        </b>
                        <span style={{ marginLeft: 8 }}>{text}</span>
                      </Box>
                    </Box>
                  );
                })
            ) : (
              <Typography color="textSecondary">{t('no_details_available')}</Typography>
            )}
          </Box>
        </DialogContent>
      </Dialog>

      <Dialog open={open} onClose={() => setOpen(false)}>
        <DialogTitle>{t('add_medical_record')}</DialogTitle>
        <DialogContent>
          <Box sx={{ mt: 2 }}>
            <TextField
              fullWidth
              label={t('condition')}
              name="condition"
              value={newRecord.condition}
              onChange={(e) => setNewRecord({ ...newRecord, condition: e.target.value })}
              margin="normal"
              required
            />
            <TextField
              fullWidth
              label={t('date')}
              type="date"
              name="date"
              value={newRecord.date}
              onChange={(e) => setNewRecord({ ...newRecord, date: e.target.value })}
              margin="normal"
              required
              InputLabelProps={{
                shrink: true,
              }}
            />
            <TextField
              fullWidth
              label={t('notes')}
              name="notes"
              value={newRecord.notes}
              onChange={(e) => setNewRecord({ ...newRecord, notes: e.target.value })}
              margin="normal"
              multiline
              rows={4}
            />
            <input
              type="file"
              accept="application/pdf,image/*"
              onChange={e => setFile(e.target.files[0])}
              style={{ marginTop: 16, marginBottom: 8 }}
            />
            <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
              <Button onClick={() => setOpen(false)}>{t('cancel')}</Button>
              <Button
                variant="contained"
                color="primary"
                onClick={handleAddRecord}
                sx={{ ml: 1 }}
              >
                {t('save')}
              </Button>
            </Box>
          </Box>
        </DialogContent>
      </Dialog>
    </Container>
  );
};

export default MedicalHistory;
