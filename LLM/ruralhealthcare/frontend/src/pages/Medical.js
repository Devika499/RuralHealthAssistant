import React, { useState, useEffect } from 'react';
import { Box, Button, TextField, MenuItem, Typography, List, ListItem, ListItemText, IconButton, Divider } from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import { useTranslation } from 'react-i18next';

const frequencies = [
  { value: 'daily', label: 'Daily' },
  { value: 'weekly', label: 'Weekly' },
  { value: 'monthly', label: 'Monthly' },
];

const Medication = () => {
  const { t } = useTranslation();
  const [medicationName, setMedicationName] = useState('');
  const [frequency, setFrequency] = useState('daily');
  const [time, setTime] = useState('09:00');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [reminders, setReminders] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Fetch reminders on mount
  useEffect(() => {
    fetchReminders();
  }, []);

  const fetchReminders = async () => {
    setLoading(true);
    setError('');
    try {
      const token = localStorage.getItem('token');
      const res = await fetch('/medications/', {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) throw new Error('Failed to fetch reminders');
      const data = await res.json();
      setReminders(data);
    } catch (err) {
      setError(err.message);
    }
    setLoading(false);
  };

  const handleAdd = async (e) => {
    e.preventDefault();
    setError('');
    try {
      const token = localStorage.getItem('token');
      const res = await fetch('/medications/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          medication_name: medicationName,
          frequency,
          time,
          start_date: startDate,
          end_date: endDate || null,
        }),
      });
      if (!res.ok) throw new Error('Failed to add reminder');
      setMedicationName('');
      setFrequency('daily');
      setTime('09:00');
      setStartDate('');
      setEndDate('');
      fetchReminders();
    } catch (err) {
      setError(err.message);
    }
  };

  const handleDelete = async (id) => {
    setError('');
    try {
      const token = localStorage.getItem('token');
      const res = await fetch(`/medications/${id}`, {
        method: 'DELETE',
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) throw new Error('Failed to delete reminder');
      fetchReminders();
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <Box sx={{ maxWidth: 500, mx: 'auto', mt: 4 }}>
      <Typography variant="h5" gutterBottom>
        {t('add_medication_reminder')}
      </Typography>
      <form onSubmit={handleAdd}>
        <TextField
          label={t('medication_name')}
          value={medicationName}
          onChange={e => setMedicationName(e.target.value)}
          fullWidth
          required
          sx={{ mb: 2 }}
        />
        <TextField
          select
          label={t('frequency')}
          value={frequency}
          onChange={e => setFrequency(e.target.value)}
          fullWidth
          required
          sx={{ mb: 2 }}
        >
          {frequencies.map(f => (
            <MenuItem key={f.value} value={f.value}>{f.label}</MenuItem>
          ))}
        </TextField>
        <TextField
          label={t('time')}
          type="time"
          value={time}
          onChange={e => setTime(e.target.value)}
          fullWidth
          required
          sx={{ mb: 2 }}
          InputLabelProps={{ shrink: true }}
        />
        <TextField
          label={t('start_date')}
          type="date"
          value={startDate}
          onChange={e => setStartDate(e.target.value)}
          fullWidth
          required
          sx={{ mb: 2 }}
          InputLabelProps={{ shrink: true }}
        />
        <TextField
          label={t('end_date_optional')}
          type="date"
          value={endDate}
          onChange={e => setEndDate(e.target.value)}
          fullWidth
          sx={{ mb: 2 }}
          InputLabelProps={{ shrink: true }}
        />
        <Button type="submit" variant="contained" color="primary" fullWidth>
          {t('add_reminder')}
        </Button>
      </form>
      {error && <Typography color="error" sx={{ mt: 2 }}>{error}</Typography>}
      <Divider sx={{ my: 4 }} />
      <Typography variant="h6" gutterBottom>
        {t('your_reminders')}
      </Typography>
      {loading ? (
        <Typography>{t('loading')}</Typography>
      ) : (
        <List>
          {reminders.map(rem => (
            <ListItem key={rem.id} secondaryAction={
              <IconButton edge="end" aria-label="delete" onClick={() => handleDelete(rem.id)}>
                <DeleteIcon />
              </IconButton>
            }>
              <ListItemText
                primary={`${rem.medication_name} (${t(rem.frequency)}, ${rem.time})`}
                secondary={`${t('from')} ${rem.start_date.slice(0,10)}${rem.end_date ? ' ' + t('to') + ' ' + rem.end_date.slice(0,10) : ''}`}
              />
            </ListItem>
          ))}
        </List>
      )}
    </Box>
  );
};

export default Medication; 